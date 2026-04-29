-- ============================================================
-- Smart Attendance System — Database Initialization
-- Run this ONLY if you need to manually create the schema.
-- (The app auto-creates tables on startup via SQLAlchemy.)
-- ============================================================

-- Connect to the attendance database before running this.

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    roll_number VARCHAR(50) UNIQUE NOT NULL,
    department VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_users_roll_number ON users(roll_number);

-- Facial embeddings table (vectors stored as bytea, FAISS handles search)
CREATE TABLE IF NOT EXISTS facial_embeddings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding BYTEA NOT NULL,
    sample_number INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON facial_embeddings(user_id);

-- Attendance logs table
CREATE TABLE IF NOT EXISTS attendance_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    attendance_date DATE NOT NULL DEFAULT CURRENT_DATE,
    check_in_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence_score FLOAT NOT NULL,
    CONSTRAINT uq_user_date UNIQUE (user_id, attendance_date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_user_id ON attendance_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs(attendance_date);

-- Verify
SELECT 'Tables created successfully!' AS status;
