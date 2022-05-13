-- Creates a trigger that resets the attribute valid_email only when email has been changed
CREATE TRIGGER change_email
BEFORE UPDATE
ON users
FOR EACH ROW
    IF (old.email <> new.email) THEN
        SET new.valid_email = 0
    END IF;
