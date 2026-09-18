CREATE DATABASE IF NOT EXISTS smartbite;
USE smartbite;

CREATE TABLE IF NOT EXISTS students (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    full_name          VARCHAR(100) NOT NULL,
    email              VARCHAR(150) NOT NULL UNIQUE,
    password_hash      VARCHAR(255) NOT NULL,
    age                INT,
    height_cm          FLOAT,
    weight_kg          FLOAT,
    gender             VARCHAR(10),
    daily_budget       DECIMAL(8,2) NULL,
    diet_preference    VARCHAR(10) NULL,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS nutrition_logs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    student_id      INT NULL,
    age             INT NOT NULL,
    gender          VARCHAR(10) NOT NULL,
    height_cm       FLOAT NOT NULL,
    weight_kg       FLOAT NOT NULL,
    activity_level  FLOAT NOT NULL,
    calories        INT NOT NULL,
    protein_g       INT NOT NULL,
    carbs_g         INT NOT NULL,
    fat_g           INT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id),
    INDEX idx_nutrition_student_time (student_id, created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS expenses (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    student_id    INT NOT NULL,
    amount        DECIMAL(8,2) NOT NULL,
    category      VARCHAR(50) NOT NULL,
    note          VARCHAR(255) NULL,
    expense_date  DATE NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id),
    INDEX idx_expenses_student_date (student_id, expense_date)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    student_id  INT NOT NULL,
    sender      ENUM('user', 'bot') NOT NULL,
    message     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id),
    INDEX idx_chat_student_time (student_id, created_at)
) ENGINE=InnoDB;