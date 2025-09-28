# Tasks: Customer Churn Prediction System

**Input**: Design documents from `/Users/kidamongus/AI7101finalproject/specs/001-implementation-specification-6/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Found: Python 3.11+, pandas/numpy/seaborn/scikit-learn/matplotlib/jupyter/MLflow
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests? ✅
   → All entities have models? ✅
   → All endpoints implemented? ✅
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- **Notebooks**: `notebooks/` for Jupyter notebooks
- **Data**: `data/` for CSV files
- **Results**: `results/` for model outputs and analysis

## Phase 3.1: Environment & Project Setup
- [x] T001 Create virtual environment and install dependencies (Python 3.11+, pandas, numpy, seaborn, scikit-learn, matplotlib, jupyter, mlflow)
- [x] T002 Create project directory structure: data/, notebooks/, src/, tests/, results/
- [x] T003 [P] Initialize MLflow tracking with sqlite database in project root
- [x] T004 [P] Create requirements.txt with all project dependencies and versions

## Phase 3.2: Contract Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T005 [P] Contract test for DataValidationContract in tests/contract/test_data_validation.py
- [ ] T006 [P] Contract test for PreprocessingContract in tests/contract/test_preprocessing.py
- [ ] T007 [P] Contract test for ModelDevelopmentContract in tests/contract/test_model_development.py
- [ ] T008 [P] Contract test for BusinessAnalysisContract in tests/contract/test_business_analysis.py

## Phase 3.3: Data Model Implementation (ONLY after contract tests are failing)
- [ ] T009 [P] Customer entity model in src/models/customer.py
- [ ] T010 [P] ChurnEvent entity model in src/models/churn_event.py
- [ ] T011 [P] FeatureSet entity model in src/models/feature_set.py
- [ ] T012 [P] ModelPerformance entity model in src/models/model_performance.py
- [ ] T013 [P] BusinessImpact entity model in src/models/business_impact.py
- [ ] T014 [P] RetentionStrategy entity model in src/models/retention_strategy.py

## Phase 3.4: Data Science Workflow Notebooks
- [ ] T015 [P] Phase 1 notebook: Problem Definition & Data Understanding in notebooks/01_problem_definition.ipynb
- [ ] T016 [P] Phase 2 notebook: Data Loading & Initial Setup in notebooks/02_data_loading_validation.ipynb
- [ ] T017 Phase 3 notebook: Data Preprocessing & Feature Engineering in notebooks/03_preprocessing_feature_engineering.ipynb (depends on T009-T011)
- [ ] T018 Phase 4 notebook: Exploratory Data Analysis in notebooks/04_exploratory_data_analysis.ipynb (depends on T017)
- [ ] T019 Phase 5 notebook: Model Development & Validation in notebooks/05_model_development_evaluation.ipynb (depends on T018)
- [ ] T020 Phase 6 notebook: Business Impact Analysis in notebooks/06_business_impact_analysis.ipynb (depends on T019)

## Phase 3.5: Core Service Implementation
- [ ] T021 Data validation service implementing DataValidationContract in src/services/data_validation_service.py
- [ ] T022 Preprocessing service implementing PreprocessingContract in src/services/preprocessing_service.py
- [ ] T023 Model development service implementing ModelDevelopmentContract in src/services/model_development_service.py
- [ ] T024 Business analysis service implementing BusinessAnalysisContract in src/services/business_analysis_service.py

## Phase 3.6: Integration & Workflow
- [ ] T025 MLflow experiment configuration and logging utilities in src/utils/mlflow_utils.py
- [ ] T026 [P] Cross-validation pipeline with SMOTE integration in src/utils/cv_pipeline.py
- [ ] T027 Feature engineering pipeline in src/preprocessing/feature_engineering.py
- [ ] T028 Model evaluation metrics and visualization utilities in src/evaluation/metrics.py

## Phase 3.7: Integration Tests
- [ ] T029 [P] End-to-end data loading and validation test in tests/integration/test_data_pipeline.py
- [ ] T030 [P] Complete preprocessing workflow test in tests/integration/test_preprocessing_pipeline.py
- [ ] T031 [P] Model training and evaluation integration test in tests/integration/test_model_pipeline.py
- [ ] T032 [P] Business impact analysis integration test in tests/integration/test_business_analysis.py

## Phase 3.8: Polish & Documentation
- [ ] T033 [P] Unit tests for utility functions in tests/unit/test_utils.py
- [ ] T034 [P] Unit tests for preprocessing functions in tests/unit/test_preprocessing.py
- [ ] T035 [P] Performance validation against 85% accuracy target in tests/performance/test_model_performance.py
- [ ] T036 [P] Executive summary generation for stakeholder presentation in src/reporting/executive_summary.py
- [ ] T037 Final validation using quickstart.md test scenarios

## Dependencies
- Environment setup (T001-T004) before everything else
- Contract tests (T005-T008) before implementation (T009-T037)
- Data models (T009-T014) before dependent notebooks (T017-T020)
- Services (T021-T024) before integration tests (T029-T032)
- Core implementation before polish (T033-T037)

## Parallel Execution Examples

### Contract Tests Phase (Run T005-T008 together):
```
Task: "Contract test for DataValidationContract in tests/contract/test_data_validation.py"
Task: "Contract test for PreprocessingContract in tests/contract/test_preprocessing.py"
Task: "Contract test for ModelDevelopmentContract in tests/contract/test_model_development.py"
Task: "Contract test for BusinessAnalysisContract in tests/contract/test_business_analysis.py"
```

### Data Models Phase (Run T009-T014 together):
```
Task: "Customer entity model in src/models/customer.py"
Task: "ChurnEvent entity model in src/models/churn_event.py"
Task: "FeatureSet entity model in src/models/feature_set.py"
Task: "ModelPerformance entity model in src/models/model_performance.py"
Task: "BusinessImpact entity model in src/models/business_impact.py"
Task: "RetentionStrategy entity model in src/models/retention_strategy.py"
```

### Integration Tests Phase (Run T029-T032 together):
```
Task: "End-to-end data loading and validation test in tests/integration/test_data_pipeline.py"
Task: "Complete preprocessing workflow test in tests/integration/test_preprocessing_pipeline.py"
Task: "Model training and evaluation integration test in tests/integration/test_model_pipeline.py"
Task: "Business impact analysis integration test in tests/integration/test_business_analysis.py"
```

## Notes
- [P] tasks = different files, no dependencies between them
- Verify contract tests fail before implementing (TDD requirement)
- MLflow experiment tracking must be configured before model development
- Seaborn is mandatory for all visualizations (constitutional requirement)
- All random seeds must be set to 42 for reproducibility
- F1-score is primary evaluation metric for imbalanced churn data
- Target accuracy: >85% with consistent cross-validation performance

## Task Generation Rules Applied
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P] (T005-T008)
   - Each contract → service implementation task (T021-T024)

2. **From Data Model**:
   - Each entity → model creation task [P] (T009-T014)
   - Relationships → dependency tracking in notebooks

3. **From Research Decisions**:
   - MLflow setup → experiment tracking tasks (T003, T025)
   - 60-day timeframe → validation in notebooks
   - Python/Jupyter → notebook creation tasks (T015-T020)

4. **From Quickstart Scenarios**:
   - Each validation test → integration test (T029-T032)
   - Environment setup → project setup tasks (T001-T004)

5. **Ordering Applied**:
   - Setup → Tests → Models → Services → Integration → Polish
   - TDD order: Contract tests before implementation
   - Constitutional compliance: Experiment tracking before modeling

## Validation Checklist
*GATE: Checked by main() before returning*

- [x] All contracts have corresponding tests (T005-T008)
- [x] All entities have model tasks (T009-T014)
- [x] All tests come before implementation (Phase 3.2 before 3.3+)
- [x] Parallel tasks truly independent (different files, no shared dependencies)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Constitutional requirements addressed (MLflow, reproducibility, business focus)
- [x] Academic requirements covered (6 phases, notebooks, presentation)

---

**Total Tasks**: 37 numbered tasks covering complete data science workflow
**Estimated Completion**: 4-5 weeks following academic timeline
**Ready for execution**: Use task IDs for systematic implementation