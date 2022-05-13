-- Creates a stored procedure AddBonus that adds a new correction for a student
CREATE PROCEDURE AddBonus (user_id INT, project_name VARCHAR(255), score FLOAT)
BEGIN
    IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
        INSERT INTO projects(name)
        VALUES (project_name);
    END IF;

    SET @project_id = (SELECT id FROM projects WHERE projects.name = project_name);

    INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, @project_id, score);
END;
