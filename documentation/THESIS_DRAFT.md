# Pneumonia AI Lab — Thesis Draft (English)

**Note for student:** Replace bracketed placeholders `[…]` with your university name, supervisor, dates, and screenshots. Page numbers in your guide may differ; align with your template. Your outline had minor numbering overlap (e.g. two “2.1” items); this draft uses a consistent logical order.

**Project summary:** Web-based **Pneumonia AI Lab** — chest X-ray classification using **DenseNet121** feature extraction, **SVM** classifier, **Grad-CAM** visualization, **Flask** backend, **SQLite** persistence, PDF reports, dashboard, analytics, and optional cloud **AI assistant** (Gemini / Groq / OpenRouter).

---

# Chapter 1 — Introduction

## 1.1 Project Introduction

### 1.1.1 Main Theme

Pneumonia remains a significant global cause of morbidity and mortality, especially where radiologist capacity is limited. Computer-aided detection (CAD) from chest radiographs can support triage and education but must never replace clinical diagnosis. This project implements **Pneumonia AI Lab**: a university-oriented demonstration system that ingests frontal chest X-ray images, computes deep features with a pretrained **DenseNet121** convolutional network (via **TensorFlow/Keras**), classifies **Pneumonia vs Normal** using a **Support Vector Machine (SVM)** trained on extracted features, and explains model attention regions using **Grad-CAM**. A **Flask** web application provides authenticated **login**, **patient-centric records**, **scan workflow**, **reports**, basic **analytics**, **PDF export**, optional **Roman Urdu / English AI assistant**, and **voice input/output** hooks in supported browsers.

The overarching theme is **responsible AI in medical imaging demos**: pairing prediction with visualization, disclaimers, user accounts, audit-friendly storage, and clear limits of automated screening.

### 1.1.2 Scope of the Project

**In scope:**
- Binary classification (**Normal / Pneumonia**) from uploaded chest X-ray images subject to demo model assumptions.
- Feature extraction backbone: **DenseNet121** (ImageNet-init or project-specific checkpoints as configured).
- Classifier: **scikit-learn SVM** on flattened feature vectors with saved **joblib** artifacts.
- **Grad-CAM** heatmap overlay for interpretability.
- Secure-style **session-based authentication**, **SQLite** persistence (`users`, `patients`, `assistant_chat`).
- **Web UI**: dashboard, new scan form, records, reports, analytics, settings, printable report layout.
- **PDF generation** for patient reports (ReportLab pipeline as implemented).
- **AI assistant**: online-only, API-key driven; patient context optionally attached where supported.

**Out of scope (typical thesis boundaries):**
- Multi-class lung pathology taxonomy (TB, COVID-, mass segmentation, etc.) unless explicitly extended.
- Clinical deployment, HIPAA/FDA/regulatory clearance, institutional PACS integration.
- Guaranteed diagnostic accuracy claims; the system is framed as **educational / research demo**.

### 1.1.3 Objectives of the Project

1. Design and implement an end-to-end **web-accessible** pneumonia screening demo with **transparent ML pipeline** (features → SVM → confidence).
2. Integrate **Grad-CAM** to visualize salient image regions and support user education.
3. Provide **secure user management** and **structured storage** of predictions, metadata, and assets.
4. Deliver **operational workflows**: scan entry, results view, reports list, analytics summaries, PDF download/print.
5. Explore **human–AI interaction** via optional **cloud LLM assistant** and **browser speech** (STT/TTS) where available.
6. Document **feasibility, requirements, design, implementation, testing, and limitations** for academic evaluation.

## 1.2 Introduction to Organization

### 1.2.1 Organizational Setup and Structure

[Describe your university / department / program: e.g. Department of Computer Science, final-year project under supervision of **Dr. Name**. Outline roles: student (analysis, design, coding, testing), supervisor (review, methodology), lab resources (GPU access if any).]

The project is executed as an **academic software engineering** deliverable: requirements → design → implementation → validation → documentation. Artefacts include source repository, SQLite schema, serialized models (`joblib`/TensorFlow checkpoints as per repo), Flask routes, frontend templates/CSS/JS, and this thesis narrative.

### 1.2.2 Main Aim and Work Environment

**Aim:** Build a reproducible prototype that showcases **clinical-style UX** around a **CNN + classical ML** hybrid, not only a Jupyter notebook pipeline.

**Work environment:**
- Development: desktop/laptop **Windows/Linux**, **Python 3.10+**, virtual environment (`requirements.txt`).
- Runtime: Flask development or production-capable server; browser-based client.
- Optional: NVIDIA GPU accelerates DenseNet inference; CPU fallback possible with latency trade-offs.

---

# Chapter 2 — System Analysis

## 2.1 Feasibility Study

### 2.1.1 Economic Feasibility

Costs are comparatively low for a student/demo scope:
- Software stack is **FOSS**: Python ecosystem, Flask, TensorFlow ecosystem (licenses per bundled components), SQLite, ReportLab Community, browser APIs.
- Optional cloud inference costs only if Gemini/Groq/OpenRouter keys are used for assistant (pay-as-you-go or free tiers depending on vendor policy at deployment time).

No mandatory proprietary PACS licence is assumed. Hardware is commodity PC/server + optional GPU.

### 2.1.2 Technical Feasibility

DenseNet121 + SVM hybrids are textbook-feasible; Grad-CAM is well-documented for TensorFlow/Keras backends. Flask + Jinja templating suffices for departmental demos. SQLite handles expected demo concurrency. Challenges (image preprocessing alignment, calibration of confidence, fairness across devices) are **manageable** with engineering discipline and scope control.

### 2.1.3 Schedule Feasibility

A typical final-year timeline can be partitioned: literature & data (2–3 weeks), model training & evaluation (3–4 weeks), web app & DB (3–4 weeks), reporting & thesis (2–3 weeks), buffer for integration/testing. **Slippage risks:** dataset curation, hyperparameter tuning, UI polish, deployment debugging.

## 2.2 Existing System: Data Gathering

### 2.2.1 Review of the Existing System

Prior art includes (a) **manual radiology reporting** only; (b) **notebook-only ML** without role-based access; (c) **commercial CAD** (different cost/compliance profile). Many academic papers stop at metrics without full software lifecycle.

### 2.2.2 Sampling and Observation

**Sampling:** Public chest X-ray datasets (e.g. NIH-style / project-specific splits as used in training scripts in repository) — student must state **exact dataset name, train/val/test split, class balance** as per their actual experiments.

**Observation:** Clinicians need **trust cues** (heatmap, confidence, printable report, disclaimers). Students need **repeatable** training and evaluation scripts.

### 2.2.3 Advantages and Disadvantages of Existing System

| Aspect | Manual / traditional | Pure research notebook | Target system (this project) |
|--------|----------------------|-------------------------|------------------------------|
| Traceability | High human judgment | Low unless versioned | Medium–high (logged predictions) |
| Throughput | Limited by staff | N/A | Batch per user session |
| Explainability | Radiologist narrative | Often absent | Grad-CAM + text fields |
| Regulation | Clinical pathway | N/A | Demo only |

## 2.3 Proposed System: Data Gathering

### 2.3.1 Proposed System

**Pneumonia AI Lab** centralizes:
- Authenticated web access to upload X-ray, capture patient metadata, run inference, store outputs, and generate reports.
- Assistant channel (optional) for educational Q&A with strict non-diagnostic framing in system prompts.

### 2.3.2 Comparison between Existing and Proposed System

**Improvements over notebook-only:** persistent records, multi-user handling, PDFs, analytics views, production-style routing and error handling.

**Limitations vs hospital PACS:** no DICOM listener, no enterprise IAM, no legal framework for patient care without certification.

## 2.4 Existing System: Data Analysis

[Insert your actual numbers: accuracy, precision, recall, F1, confusion matrix, ROC-AUC if computed. Describe train/validation methodology, cross-validation if any, and failure cases (poor contrast, foreign objects, pediatric vs adult if applicable).]

The proposed system’s **analysis module** should reference the same metrics used in **Chapter 4** evaluation tables.

## 2.5 Requirement Specifications

### 2.5.1 Functional Requirements

1. **User registration & login** with session management.
2. **Upload chest X-ray** (supported formats as per app configuration).
3. **Run prediction** and display label + confidence.
4. **Show Grad-CAM** vs original compare view where implemented.
5. **Save patient record** with timestamp and creating user.
6. **List/search/filter** records and open detail view.
7. **Reports** grid with PDF download / print-friendly view.
8. **Analytics** charts/summaries driven from stored predictions.
9. **Settings**: environment hints for API keys (non-secret surfaces only in UI).
10. **Assistant** chat with history persistence (SQLite `assistant_chat`).

### 2.5.2 Non-Functional Requirements

- **Usability:** responsive dashboard, readable light/dark themes.
- **Performance:** bounded image dimensions; inference within acceptable latency on target hardware [state seconds].
- **Reliability:** graceful failure when models missing / keys absent.
- **Maintainability:** modular Flask blueprint-style structure encouraged; documented `requirements.txt`.
- **Privacy:** credential storage pattern [hashed passwords if implemented — verify codebase]; disclaimers on educational use.

## 2.6 Safety Requirements

- Prominent disclaimer: **Not a substitute for licensed medical diagnosis.**
- Input validation & file-type checks mitigating malicious uploads.
- Rate limiting recommended if publicly exposed ([optional deployment note]).
- Avoid storing unnecessary PHI in demo deployments.

## 2.7 Security Requirements

- **Authentication** middleware on protected routes.
- Session cookies with appropriate flags in production (**HTTPS**, `Secure`, `HttpOnly`, `SameSite`).
- Secrets (API keys) via **environment** / `.env` (never committed publicly).
- SQL injection mitigation via parameterized queries (SQLite placeholders in codebase pattern).

## 2.8 Deliverables

1. Runnable **source code** repository.
2. **Trained artefacts** (`joblib` SVM / TensorFlow model files) + preprocessing contract.
3. **SQLite schema** (`users`, `patients`, `assistant_chat`).
4. **User documentation / thesis** chapters.
5. **Demo deployment** instructions (README equivalent).
6. **Test logs** / screenshots appendix.

---

# Chapter 3 — System Design

## 3.1 Introduction to System Design

Architecturally, the solution follows classic **three-tier thinking** adapted to Flask:
- **Presentation:** HTML templates, CSS (`style.css`, `dashboard.css`, `healthcare-ui.css`), JavaScript (theme, assistant, confirm modals, voice).
- **Application:** Flask routes (`app.py`) orchestrating auth, inference, PDF, chat API.
- **Data:** SQLite file `database/app.db`.

## 3.2 Proposed System and its Features

### 3.2.1 Features of the Proposed System

- Role-scoped **patient records** linked to `created_by_user`.
- **Grad-CAM** educational overlay.
- **Analytics** summary of historical predictions.
- **Internationalization-friendly** Roman Urdu strings in assistant UX.
- **Accessibility touches:** skip links, ARIA on chat log (where present).
- **Voice:** Web Speech API for dictation and optional TTS.

## 3.3 System Design Using UML

[Insert figures; below is textual specification for drawing tools.]

### 3.3.1 Use Case Diagram 1 — Authentication

**Actors:** Visitor, Registered User, System.  
**UCs:** Register, Login, Logout, Reset password [if implemented].

### 3.3.2 Use Case Diagram 2 — Radiologist / Clinician Demo User

**UCs:** Create new scan, View prediction, View Grad-CAM, Save record, Export PDF, Print report.

### 3.3.3 Use Case Diagram 3 — Administrator / Power User (optional)

**UCs:** View analytics, Manage environment configuration [scope to settings page].

### 3.3.4 Use Case Diagram 4 — AI Assistant

**UCs:** Ask question, Receive answer, Clear history (with confirm), Optional voice I/O.

## 3.4 Activity Diagram

### 3.4.1 Activity Diagram 1 (User Flow)

Login → Dashboard → Choose “New Scan” → Fill patient info → Upload image → Submit → View result + heatmap → Save / Download PDF → Logout.

### 3.4.2 Activity Diagram 2 (System Processing Flow)

Receive HTTP POST → Validate session → Store temp image → Preprocess → DenseNet feature forward → SVM decision → Confidence mapping → Compute Grad-CAM → Persist DB rows → Respond HTML/JSON fragments.

## 3.5 Sequence Diagram

### 3.5.1 Sequence Diagram 1 (User Interaction Flow)

User, Browser, Flask, Inference Service (in-process), DB: message lifelines for `/detection` submit round-trip.

### 3.5.2 Sequence Diagram 2 (System Processing Flow)

Emphasizes TensorFlow graph execution, CAM map resize, PNG write, DB commit ordering.

## 3.6 Data Flow Diagrams (DFD)

### 3.6.1 DFD Level 0 (Context Diagram)

**External entities:** Radiologist/User, Administrator (optional), Cloud LLM Gateway (optional).  
**Central process:** Pneumonia AI Lab Web System.  
**Data stores:** Local Model Store, SQLite DB.

### 3.6.2 DFD Level 1 (Detailed Diagram)

Processes: Authenticate; Manage Patients; Inference Engine; Reporting; Analytics Aggregator; Assistant Proxy.

## 3.7 Database Design

### 3.7.1 Entities of the System

- **User:** `{ id, username, email, password }`
- **Patient:** `{ id, patient_name, age, gender, phone, address, image_path, gradcam_path, result, confidence, description, suggestions_json, prediction_date, created_by_user }`
- **AssistantChat:** `{ id, username, role, content, created_at }`

## 3.8 Database Structure

[Include SQL `CREATE TABLE` excerpts from implementation — copy from `database.py` verbatim in final thesis with caption.]

## 3.9 Entity Relationship Diagram

### 3.9.1 Advantages of ER Diagram

Clarifies **1:N** relationship between `User` and `Patient` via `created_by_user`, and **1:N** chat messages per `User` in `assistant_chat`.

### 3.9.2 ER Diagram

[Draw: USER 1—* PATIENT; USER 1—* ASSISTANT_CHAT.]

---

# Chapter 4 — System Development & Implementation

## 4.1 Introduction to System Development & Implementation

Implementation followed iterative coding: baseline inference script → Flask integration → persistent storage → UI themes → assistant & voice layers.

## 4.2 Tool/Language Selection

### 4.2.1 Client-Side Technology

- HTML5, CSS3, progressive enhancement JavaScript.
- Fetch API for `/api/chat` JSON exchange.
- Web Speech API (Chrome/Edge preferred).

### 4.2.2 Server-Side Technology

- **Python**, **Flask ≥3.0** microframework.
- **Jinja2** templating (`frontend/template`).
- Static assets under `frontend/static`.

### 4.2.3 AI & Supporting Libraries

- **TensorFlow** (Keras Applications DenseNet121).
- **scikit-learn** SVM (`joblib` persistence).
- **OpenCV** / imaging utilities for preprocessing overlays.
- **NumPy**, **Pillow**.
- Optional **google-generativeai**, cloud REST via **urllib** abstraction in `chatbot.py`.

### 4.2.4 Development Tools

- Git version control.
- IDE: VS Code / PyCharm / Cursor.
- Debugging with Flask `--debug`, browser DevTools Network tab.

## 4.3 Hardware for the System

**Minimum:** multi-core CPU, 8–16 GB RAM, SSD.  
**Recommended:** NVIDIA GPU ≥ 8 GB VRAM for comfortable DenseNet throughput. Client devices: modern evergreen browsers.

## 4.4 Software Coding

[Summarize notable modules:]  
`backend/app.py` — routing, inference orchestration.  
`backend/database.py` — persistence.  
Optional model utilities / training notebooks [reference paths in repo].

**Snippet strategy:** cite 15–25 lines max per appendix for ethical brevity; full listing in appendix CD/USB per university rule.

### Key algorithms (pseudo)

1. `img = preprocess(upload)` → tensor batch.
2. `features = backbone.predict(img)`
3. `label = svm.predict(features)` ; `confidence = calibrated decision function / probability layer` [match actual].
4. `heatmap = grad_cam(model, tensor, predicted_class)`

## 4.5 Testing of Program

### 4.5.1 Testing Strategies

- **Unit-style:** DB insert/select, prediction on fixed fixture image checksum.
- **Integration:** Flask test client POST login + scan flow [if pytest present — else manual].
- **UI:** cross-browser sanity (Chrome baseline).
- **Security:** naive password policy review; session fixation awareness [document improvements].

## 4.6 Implementation Phase

### 4.6.1 Methods of Implementation

1. **Direct Implementation:** Big-bang rollout acceptable for demo; production would use phased pilot.
2. **Steps of Implementation:** Environment bootstrap → migrations → routing → frontend polish → inference packaging → docs.
3. **System Deployment:** `python backend/app.py` or WSGI server (`gunicorn`/`waitress` recommended for production behind HTTPS reverse proxy `[name Nginx]`).
4. **Benefits of Implementation:** Unified educational platform consolidating model + UX + reproducibility artefacts.

---

# Chapter 5 — User's Guide

> Replace each subsection with cropped screenshots labelled **Fig 5.x**.

## 5.1 Screenshot of System

Provide **overall dashboard** annotated figure.

## 5.2 Main Page (Login Page)

Steps: Navigate URL → Enter username/password → Submit → Landing dashboard. **[Insert screenshot]**

## 5.3 Page 2 (Signup Page)

Explain field validation errors, password rules if any. **[Insert screenshot]**

## 5.4 Page 3 (Project Page / Prediction Page)

Clarify in thesis that your TOC label maps to implemented **Detection / New Scan** route: Upload X-ray → patient demographics → Predict → overlays. **[Insert screenshot]**

### Additional recommended pages for completeness

- Patient Records listing & row actions  
- Patient detail report + Compare slider  
- Reports grid + share/PDF workflows  
- AI Assistant + microphone / auto-speak toggle  
- Analytics & Settings  

---

# Chapter 6 — Conclusion & Future Work

## 6.1 Conclusion

The project demonstrates a **CNN feature + SVM** hybrid pneumonia screening demo packaged in a **maintainable web stack** with **Grad-CAM** interpretability.

### 6.1.1 The Hurdles

- Dataset imbalance & domain shift.  
- GPU availability vs CPU timeouts.  
- Browser speech inconsistencies.  
- Cloud assistant compliance & hallucination safeguards.

### 6.1.2 Further Future Plan

- Multi-label pathology expansion.  
- DICOM ingestion + anonymization pipeline.  
- Model calibration & uncertainty metrics.  
- Federated learning experimentation (stretch goal).

### 6.1.3 Successful Achievement

Delivered integrated **prediction + persistence + UX + explanations + reporting** aligning capstone objectives.

### 6.1.4 Final Words

Responsible adoption requires **clinical validation**, governance, and continuous monitoring—but educational prototypes catalyze necessary literacy in medical AI tooling.

---

# References

1. Huang, G., et al. **Densely Connected Convolutional Networks.** CVPR, 2017.  
2. Selvaraju, R.R., et al. **Grad-CAM: Visual Explanations via Gradient-based Localization.** ICCV, 2017.  
3. NIH Chest X-Ray dataset publications / CheXNet reference as applicable [verify exact citations you used].  
4. Pedregosa et al., **Scikit-learn: Machine Learning in Python.** JMLR, 2011.  
5. Flask Documentation. https://flask.palletsprojects.com/

# Bibliography

[Add supplementary reading: WHO pneumonia fact sheets; radiology reporting standards; HIPAA high-level primer if comparing compliance; HTML5 Speech APIs documentation.]

---

## Appendix Checklist for Your Institution

| Item | Done? |
|------|-------|
| Plagiarism report | |
| Supervisor approval signatures | |
| Code listing CD | |
| Test case tables | |

---

*End of draft — expand each subsection to target page lengths per your department.*
