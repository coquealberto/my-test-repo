# Pipeline para Clasificación de Documentos Jurídicos

## 0) Objetivo y Arquitectura

*   **Dominio y Longitud:** Documentos jurídicos en español, largos (≈1.5k tokens de media), con señales semánticas distribuidas en secciones.
*   **Meta:** macro-F1 ≥ 0.90–0.95, estable por clase (no solo micro-F1).
*   **Estrategia:**
    *   **DAPT (Domain-Adaptive Pre-Training):** Especializar el encoder en el dominio legal.
    *   **Fine-tuning:** LoRA + chunking + pooling jerárquico (tokens → chunk → documento).
    *   **Ensemble:** TF-IDF+SVM para sumar el "pilar léxico" a la semántica contextual.
*   **Justificación:** Ataca tres frentes: léxico, semántica, estructura larga.

## 1) Preparación de Datos y Splits

*   **Normalización:** Limpia cabeceras/pies repetidos, corrige encoding, conserva entidades clave.
*   **Etiquetado:** 35 etiquetas (+1 en tus CSVs). Mantener guía de etiquetado.
*   **Split:** Estratificado por documento (no por chunk): 80/10/10 o 70/15/15. Semilla fija.
*   **Metadatos:** doc_id, label, longitudes, fuente, fecha.
*   **Por qué el split por documento:** Evita leakage.

## 2) Baseline TF-IDF + SVM (Pilar Léxico)

*   **Vectorizador:** n-grams 1–3, min_df moderado, sublinear_tf, use_idf.
*   **SVM:** Lineal con C ajustado por CV; guardar probabilidades (predict_proba) por clase.
*   **Por qué:** Capta frases y fórmulas recurrentes. Fuerte en clases con patrones rígidos. Calibración adicional para el ensemble.

## 3) DAPT (Domain-Adaptive Pre-Training) - `01_dapt_pretrain.py`

*   **Qué hace:**
    *   Carga tokenizador y modelo MLM (AutoTokenizer + AutoModelForMaskedLM) desde un modelo base (Longformer 4096 o RoBERTa-BNE).
    *   Carga corpus jurídico no etiquetado desde `data/processed_tess_70k_cleaned` (.txt envueltos en `datasets.Dataset`).
    *   Tokeniza en paralelo (num_proc = `os.cpu_count() // 2`) con truncation y `max_length`.
    *   Prepara `DataCollatorForLanguageModeling` (MLM) con `mlm_probability=0.15`.
    *   Define `TrainingArguments`: `max_steps`, `learning_rate=5e-5`, `warmup_ratio=0.05`, `weight_decay=0.01`, mixed precision, `save_steps`, `save_total_limit`.
    *   Lanza `Trainer.train()` y guarda modelo+tokenizer en `experiments/dapt_mlm_local`.
*   **Justificación:**
    *   Alinea el espacio semántico con el dialecto jurídico.
    *   MLM es barato y efectivo.
    *   `num_proc` acelera el mapping.
    *   Mixed precision y grad accumulation logran batch efectivo alto sin desbordar VRAM.
*   **Salida:** Checkpoint en `experiments/dapt_mlm_local/checkpoint-XXXX`.

## 4) Fine-tuning con LoRA + Chunking + Pooling Jerárquico - `02_lora_finetune.py`

### 4.1 Dataset y Carga de Textos

*   CSV con `markdown_path` y `label` (luego `label_id` via `LabelEncoder`).
*   `DocsDataset`:
    *   Lee el texto completo del `.md`.
    *   (Opcional) Pre-pend del procedimiento: `"[PROCEDIMIENTO: X]"`.
    *   Chunking del texto con un `Chunker` propio.
*   **Por qué:** Leer desde `.md` conserva estructura; el prepend añade un "feature" simbólico.

### 4.2 Chunker (tokens → chunks)

*   Tokeniza sin `add_special_tokens` y divide en trozos de longitud fija (`chunk_len`) con solape (`stride`).
*   Añade especiales con `tokenizer.build_inputs_with_special_tokens(...)`.
*   Respeta `tokenizer.model_max_length` (clamp).
*   **Por qué:**
    *   Solape recupera señales en los bordes de los chunks.
    *   Reservar 2 tokens para `[CLS]/[SEP]` evita desbordes.

### 4.3 Collator (empaquetado a lotes)

*   Toma la lista de chunks por documento y:
    *   Trunca a `max_chunks` (control de VRAM).
    *   Calcula `max_c` (chunks máximos en el lote) y `max_l` (longitud máxima de chunk en el lote), con clamp a `model_max_length`.
    *   Pad tokens dentro de cada chunk a `max_l` y pad chunks por documento a `max_c`.
    *   Devuelve tensores `input_ids [B, C, T]`, `attention_mask [B, C, T]`, `labels [B]`.
*   **Por qué:**
    *   Evita listas desiguales.
    *   El clamp mata el warning de "sequence length too long".
    *   Mantener la dimensión de chunks `C` explícita permite pooling jerárquico.

### 4.4 Arquitectura del Modelo (encoder + pooling jerárquico + head)

*   **Encoder:**
    *   `AutoModel.from_pretrained(MODEL_NAME)` carga el encoder del checkpoint DAPT.
    *   LoRA con `peft`:
        *   `TaskType.FEATURE_EXTRACTION`
        *   `r`, `alpha`, `dropout` (típico `r=16–32`, `alpha=16–32`).
        *   `target_modules`:
            *   RoBERTa/BERT: `"query","key","value"`.
            *   Longformer: `"self_attn.q_proj","self_attn.v_proj"`.
            *   GPT-like: `"q_proj","v_proj"`.
        *   **Razón:** LoRA inyecta "bajas-ranks" en proyecciones de atención.
*   **Pooling Jerárquico:**
    *   `TokenMaskedMean`: Promedio sobre tokens dentro de cada chunk (enmascarado por `attention_mask`) → produce `[B, C, H]`.
    *   **Por qué:** Reduce ruido a nivel de tokens.
    *   `ChunkAttentionPooling`: Atención sobre chunks para obtener `[B, H]`.
    *   **Por qué:** Pondera secciones más útiles.
    *   Dropout + `Linear(H→num_labels)` → logits por documento.
*   **Camino Directo de Datos:**
    *   Entradas `[B, C, T]` se "aplanan" a `[B*C, T]` para pasar por el encoder.
    *   Se re-forma a `[B, C, T, H]`, se hace mean por tokens → `[B, C, H]`, se calcula máscara de chunk y se aplica atención por chunk → `[B, H]`, y finalmente clasificación → `[B, num_labels]`.
*   **Por qué esta elección:**
    *   Con docs ~1500 tokens, tendrás ~3–4 chunks de 512 (RoBERTa).
    *   La atención jerárquica suele superar "mean sobre todo".
    *   Evitar el pooler no inicializado (warning de Roberta) es intencional.

### 4.5 Entrenamiento: pérdidas, pesos, métricas, early stopping

*   **Pérdida con Pesos de Clase:**
    *   `compute_class_weight('balanced', ...)` para `class_weights`.
    *   `WeightedLossTrainer` (subclase de `Trainer`) que:
        *   Usa `nn.CrossEntropyLoss(weight=class_weights)`.
        *   Mueve los pesos a la misma device que los logits dentro de `compute_loss`.
        *   (Opcional) Label smoothing 0.05.
*   **Métricas:**
    *   `compute_metrics` devuelve `accuracy`, `precision`, `recall`, F1 macro.
    *   `metric_for_best_model="eval_f1"` y `greater_is_better=True`.
*   **TrainingArguments:**
    *   `eval_strategy="steps"`, `eval_steps=EVAL_STEPS` y `save_steps=EVAL_STEPS`.
    *   `load_best_model_at_end=True`.
    *   `warmup_ratio=0.06` y LR modestos (2e-5).
    *   `remove_unused_columns=False`.
    *   `bf16/fp16`.
    *   `gradient_accumulation_steps`.
*   **Early Stopping:**
    *   `EarlyStoppingCallback(patience=3)`: Corta cuando el macro-F1 deja de mejorar.

### 4.6 Evaluación y Guardado

*   `trainer.predict(test_ds)` → Predicciones por documento.
*   `classification_report` (per-label, macro, micro).
*   Guardas:
    *   `final_model` (pesos del encoder+LoRA y la cabeza).
    *   `tokenizer`.
    *   `labels.csv` (orden de clases del `LabelEncoder`).

## 5) Problemas Detectados en Logs y Soluciones

*   `LoraConfig.__init__() got an unexpected keyword argument 'lora_type'`: API antigua. Usar `TaskType.FEATURE_EXTRACTION` y configurar `LoraConfig` sin `lora_type`.
*   `Batch size mismatch (4096 vs 8)`: Pooling incorrecto. Pooling jerárquico que colapsa a `[B, H]`.
*   `CUDA vs CPU en loss`: `class_weights` en CPU. Mover los pesos a la device de logits dentro de `compute_loss`.
*   `Warning “Token indices sequence length is longer than … (4387 > 4096)”`: El `Chunker` y el `Collator` clamp y recortan cada chunk.

## 6) Cómo Esto Lleva a macro-F1 0.90–0.95

*   **DAPT:** El encoder ya ha visto masivamente español jurídico.
*   **Chunking + pooling jerárquico:** Capta contexto largo y pondera secciones informativas.
*   **LoRA:** Adaptación con pocos parámetros.
*   **Class weights + label smoothing:** Estabiliza minoritarias y calibra probabilidades.
*   **Early stopping por macro-F1:** Alineado con tu métrica objetivo.

## 7) Palancas de Mejora

*   **Top-k chunks:** Seleccionar los k más "salientes" (por TF-IDF o "auto-atención" media).
*   **Diferenciar LR:** Encoder: 1e-5, Head + chunk-attention: 5e-5.
*   **Freeze parcial (warmup):** Congela primeras 6–9 capas por 0.5–1 época.
*   **Gradient checkpointing:** Habilítalo si quieres subir `CHUNK_LEN`.
*   **QLoRA (4-bit):** Si quieres probar modelos más grandes o guardar más VRAM.
*   **Focal loss:** Si detectas clases "duramente minoritarias".
*   **Scheduler:** "cosine" con warmup a veces funciona mejor que linear.
*   **EMA (Exponential Moving Average) de pesos:** Suaviza oscilaciones.

## 8) Ensemble con TF-IDF + Encoder

*   **Features:** `p_tfidf ∈ R^36`, `p_encoder ∈ R^36`.
*   **Meta-modelo:** Logistic Regression / LightGBM con validación estratificada.
*   **Clave:** Usar predicciones out-of-fold en train para no fugas.
*   **Por qué:** TF-IDF captura patrones rígidos, el encoder aporta semántica.

## 9) Validación, Reproducibilidad y Operación

*   Test "sellado": Evalúa una sola vez tras seleccionar hiperparámetros con val.
*   Semillas fijas y guardar `LabelEncoder` (`labels.csv`).
*   `remove_unused_columns=False`.
*   Reanudar (`resume_from_checkpoint`) si se corta.
*   Ejecución en servidor: nohup/tmux + logs.

## 10) Qué Hace Cada Bloque del Script (Resumen Conceptual)

*   **Imports:** HF Transformers, PEFT LoRA, sklearn, PyTorch, pandas, numpy.
*   **Chunker:** Tokeniza sin especiales, parte por `chunk_len/stride`, añade especiales.
*   **Pooling:**
    *   `TokenMaskedMean`: Mean por tokens (enmascarado) → `[B,C,H]`.
    *   `ChunkAttentionPooling`: Atención sobre chunks → `[B,H]`.
*   **DocClassifier:**
    *   Carga encoder (DAPT).
    *   LoRA (task=`FEATURE_EXTRACTION`, `target_modules`).
    *   Pooling jerárquico + Dropout + Linear.
    *   `forward`: `[B,C,T]` → `[B*num_chunks,T]` → encoder → reshape → token-mean → chunk-attention → logits `[B,num_labels]`.
*   **DocsDataset:** Lee texto, aplica Chunker, codifica label.
*   **DocCollator:** Construye batch 3D con pad de tokens y chunks.
*   **WeightedLossTrainer:**
    *   `__init__`: Guarda los pesos y crea `CrossEntropyLoss` (con label smoothing).
    *   `compute_loss`: Mueve `weight` a la device de logits; calcula loss.
*   **compute_metrics:** Calcula accuracy, macro-precision/recall/F1.
*   **TrainingArguments:**
    *   LR, batch, grad accumulation, epochs, warmup, weight decay.
    *   `eval_strategy`, `eval_steps`.
    *   `load_best_model_at_end`, `metric_for_best_model`.
    *   `BF16/FP16`.
    *   `remove_unused_columns`.
*   **Entrenamiento:** `trainer.train()`.
*   **Evaluación:** `trainer.predict(test_ds)` → `classification_report`.
*   **Guardado:** `trainer.save_model`, `tokenizer.save_pretrained`, `labels.csv`.

## 11) Checklist Práctico

*   Confirmar `target_modules` según encoder real.
*   `CHUNK_LEN/STRIDE`: `chunk_len=512`, `stride=128–192`.
*   `MAX_CHUNKS`: 8–12 si hay VRAM.
*   Label smoothing: pon 0.05 en `CrossEntropyLoss`.
*   Class weights: mantén `balanced`; si siguen mal, prueba focal loss.
*   Eval por pasos + early stopping por macro-F1.
*   Guardar `predict_proba` del SVM para el ensemble.

Con todo esto, tienes un pipeline sólido y explicable.
