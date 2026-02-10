-- Snowflake-style schema for University Bulk Order & Predictive Procurement Analytics

-- FACT TABLE: ENROLLMENTS
CREATE TABLE IF NOT EXISTS ENROLLMENTS (
    enrollment_id        BIGINT PRIMARY KEY,
    sis_user_id          BIGINT NOT NULL,
    section_id           BIGINT NOT NULL,
    enrollment_date      DATE,
    fee_included_flag    BOOLEAN,
    actual_purchase_flag BOOLEAN, -- target label
    predicted_purchase_prob FLOAT, -- model output (optional)
    term_code            VARCHAR(16),
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DIMENSION: STUDENT_MASTER
CREATE TABLE IF NOT EXISTS STUDENT_MASTER (
    sis_user_id        BIGINT PRIMARY KEY,
    financial_condition VARCHAR(32),
    region              VARCHAR(64),
    gpa                 NUMERIC(3,2),
    commuter_distance_km NUMERIC(6,2),
    scholarship_type    VARCHAR(64),
    housing_status      VARCHAR(32)
);

-- DIMENSION: ADOPTIONS (Product Catalog)
CREATE TABLE IF NOT EXISTS ADOPTIONS (
    isbn               VARCHAR(32) PRIMARY KEY,
    title              VARCHAR(256),
    publisher          VARCHAR(128),
    retail_new_price   NUMERIC(10,2),
    retail_rent_price  NUMERIC(10,2),
    is_digital         BOOLEAN,
    is_required        BOOLEAN,
    course_level       VARCHAR(16)
);

-- DIMENSION: SECTIONS (Context)
CREATE TABLE IF NOT EXISTS SECTIONS (
    section_id   BIGINT PRIMARY KEY,
    course_code  VARCHAR(32),
    dept_code    VARCHAR(16),
    term_year    VARCHAR(16),
    instructor   VARCHAR(128),
    modality     VARCHAR(32) -- online, hybrid, in-person
);

-- RELATIONSHIPS
-- ENROLLMENTS.sis_user_id   -> STUDENT_MASTER.sis_user_id
-- ENROLLMENTS.section_id    -> SECTIONS.section_id
-- SECTIONS.course_code + ADOPTIONS.isbn define the adopted material per section.

