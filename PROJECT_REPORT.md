# Detailed Project Documentation: Predictive Procurement Analytics System

## 1. Executive Summary
The **Predictive Procurement Analytics System** is a sophisticated, end-to-end data mining and machine learning application tailored for university textbook procurement. Universities historically over-order textbooks using naive “100% of enrollment” rules, leading to immense excess inventory costs, or under-order, creating stockouts. 

This system solves the over-ordering problem by analyzing real transactional and demographic data to forecast the exact probability that an individual student will purchase a course bundle or opt-out. By exposing these predictions through a highly interactive Streamlit dashboard, procurement officers can optimize inventory, negotiate publisher pricing selectively, and adapt to digital/physical format shifts.

---

## 2. System Architecture
The application is structured into four primary logical layers mapping directly to the codebase:
1. **Database / Schema Layer (`schema.sql`)**: Defines the target structural environment for the facts and dimensions.
2. **Data Ingestion & ETL Layer (`etl_pipeline.py`)**: Responsible for reading large-scale, fragmented raw CSV data, safely sampling it, and mapping thousands of raw data points to meaningful analytical features.
3. **Machine Learning Engine (`feature_engine.py`)**: Consumes the ETL features, trains a Random Forest Classifier in real-time or offline, and computes individual purchase probabilities and feature importances.
4. **Decision Support Dashboard (`dashboard_app.py`)**: A Streamlit web application providing executives with interactive KPIs, risk distributions, and behavioral insights based entirely on the ML model's outputs.

---

## 3. Data Ingestion & ETL Layer (`etl_pipeline.py`)

### 3.1 Raw Data Sources
The system was recently upgraded from a synthetic data mockup to reading massive, real-world data files. The data originates from the `new/Augmented/` directory, which contains large, balanced dataset CSVs split by university (e.g., FIU, EKU, Mercer, Campbell). 

### 3.2 Memory-Safe Ingestion
Because compiling multiple gigabytes of CSV data simultaneously in Pandas would cause an Out-Of-Memory (OOM) crash in a standard web environment, the ETL pipeline employs **fractional sampling**:
- `glob` identifies all matched CSV target files.
- The pipeline utilizes `df.sample(frac=sample_frac, random_state=seed)` natively. By defaulting to a fraction (e.g., 3%), the system extracts an operationally perfect, statistically significant subset (roughly 410,000 rows) allowing the model to train and the dashboard to render instantaneously.

### 3.3 Minute Feature Mapping & Engineering
Every column from the raw dataset is carefully mapped to the semantic layer expected by the ML model and Dashboard.

**Categorical Mappings:**
- **`Term`**: Synthesized by concatenating the raw `term_code` (e.g., "F", "W") and `term_year` (e.g., "21", "22"). 
- **`Dept_Code`**: Mined dynamically by splitting the verbose `section_id` string (e.g., "BN-8304-1-126...") and extracting the 4th token representing the department.
- **`Student_Type`**: Safely mapped from `student_full_part_time_status` ("F" mapped to "Full-Time", "P" to "Part-Time").
- **`Format`**: Based strictly on the `ebook_ind`. A `1.0` binary flag converts to "Digital"; otherwise, it defaults to "Physical".

**Economic Feature Engineering:**
- **`Rental_to_Retail_Ratio`**: Calculates (`retail_new_rent` / `retail_new`). If a book costs $100 natively but relies on a $50 rental fee, the ratio is 0.5. Values are clipped securely between `0.0` and `1.5` to avoid zero-division errors.
- **`Arbitrage_Index`**: Calculated mathematically as `1.0 - Rental_to_Retail_Ratio`. This captures the primary economic incentive for a student to opt-out. A high Arbitrage Index means the student saves massive amounts of money by avoiding the official bundle.
- **`Wallet_Pressure_Score`**: Normalizes the raw `price_affordability_score` across a `0.0` to `1.0` spectrum, quantifying how much financial strain a student carries in a given semester.
- **`Digital_Lock_Flag`**: Direct pass-through of `ebook_ind`. Conceptually, physical books can be circumvented via second-hand markets. Digital access codes cannot, meaning a `1.0` digital lock practically forces an opt-in.

**Labels:**
- **`Actual_Purchase_Flag`**: Derives from the absolute truth column `will_buy`. A `1` means they kept the bundle; `0` means they opted out.
- **`Opt_Out_Probability`**: For baseline tracking, statically set to `1.0 - Actual_Purchase_Flag`.

---

## 4. Machine Learning Engine (`feature_engine.py`)

### 4.1 The Model
The predictive engine utilizes Scikit-Learn's `RandomForestClassifier`.
- **Hyperparameters:** `n_estimators=50` (50 distinct decision trees), `max_depth=5` (to prevent overfitting on the noise).
- **Execution:** Uses `n_jobs=-1` to parallelize training across all available CPU cores perfectly.

### 4.2 The Features
The model is strictly trained on seven explicit, un-biased predictors:
1. `Arbitrage_Index`
2. `Wallet_Pressure_Score`
3. `Digital_Lock_Flag`
4. `Rental_to_Retail_Ratio`
5. `family_annual_income` (Income brackets dictating price threshold sensitivities)
6. `is_rental` (Binary flag declaring if the adoption is fundamentally a rented item)
7. `has_scholarship` (Students with external funding behave drastically differently regarding book purchases)

### 4.3 Predictions and Projections
Once trained on the cleanly dropped and mapped `train_df`, the model forecasts `.predict_proba()` across the absolute entire dataset. 
- It isolates the exact probability class corresponding to a `1` (Actual Purchase) and binds it to the target column `Predicted_Purchase_Prob`.
- **Financial Projection Engine**: It calculates total capital risk row-by-row via the formula: `Projected_Spend = Predicted_Demand_Units * Unit_Price * Predicted_Purchase_Prob`. 
- **Explainability**: The code explicitly pulls `clf.feature_importances_` to generate a lightweight DataFrame mapping each of the 7 features to their relative predictive dominance.

---

## 5. Decision Support Dashboard (`dashboard_app.py`)

The user interface wraps the entire pipeline within an interactive, glassmorphism-themed dark-mode Streamlit app utilizing Plotly.

### 5.1 Caching Strategy
Because data parsing and Random Forest training take massive computational resources, the `get_data()` function is explicitly decorated with `@st.cache_data`. This guarantees that the 10-20 second initialization delay occurs exactly **once** upon boot. All subsequent UI interactions (sliders, filters) are handled in sub-millisecond memory execution.

### 5.2 Filter Bar (Left Panel)
Dynamically populated directly from the dataset's unique values, allowing executives to slice the data by:
- Term
- Department Code
- Publisher
- Student Type (Full vs Part time)
- Format (Digital vs Physical)
When a filter changes, a Pandas boolean masking strategy seamlessly isolates the sub-population.

### 5.3 KPI Executive Row
Four massive metric cards aggregate the filtered dataset:
1. **Total Predicted Demand**: Summation of expected units.
2. **Total Projected Spend**: Converted tightly into a Millions ($M) float format for immediate university budget reconciliation.
3. **Digital vs Physical**: Calculates the dynamic percentage swing of physical textbooks transitioning to digital formats.
4. **High-Risk Opt-Out Rate**: A hard metric defining the exact percentage of the population holding an opt-out probability cleanly exceeding 60%.

### 5.4 Feature Graphs
- **Price Sensitivity Scatter**: A Plotly scatter graph explicitly tracking `Rental_to_Retail_Ratio` against `Opt_Out_Probability`, color-coded uniquely by `Dept_Code`. Identifies thresholds where increasing markup rapidly triggers mass student abandonment.
- **Model Explainability**: Renders a horizontal bar chart directly consuming the `fi_df` tuple outputted by the ML model. It proves exactly *why* the model is answering the way it is (e.g., proving whether `Digital_Lock_Flag` outweighs `family_annual_income`).
- **Format Preference & Department Risk**: Grouped bar charts dictating format preferences by student demographics, paired closely with an aggregated tracker isolating the strictly top 5 highest opt-out risk departments campus-wide.

---

## 6. Target Database Schema (`schema.sql`)
While currently powered heavily via CSV, the structural goal explicitly models a Snowflake layout defining the end-state data warehouse targets:
- **`ENROLLMENTS` Fact Table**: Captures raw target outcomes (`actual_purchase_flag`, `predicted_purchase_prob`).
- **`STUDENT_MASTER` Dimension Table**: Anchors immutable demographics (`financial_condition`, `commuter_distance`).
- **`ADOPTIONS` Dimension Table**: Outlines product pricing thresholds and cover format definitions.
- **`SECTIONS` Dimension Table**: Traces the courses down to the specific instructor and modality mappings.

---

## 7. Operational Workflow
To execute the entirely rebuilt application:
1. Navigate to the root directory `/home/iiitl/Documents/DATA mining project/`
2. Make sure the virtual environment is enabled: `source .venv/bin/activate`
3. Launch the dashboard server: `streamlit run dashboard_app.py`
4. The system will recursively hunt the `new/Augmented/*.csv` files, isolate the balanced dataset subsets, sample them, train the 50-tree Random Forest, and open `localhost:8501`. 

All code is deliberately modularized; modifying data parsing implies editing `etl_pipeline.py`, while updating prediction thresholds is cleanly isolated in `feature_engine.py`.
