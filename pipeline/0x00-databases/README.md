# Databases

## Resources

* [SQL Operators](https://www.w3schools.com/sql/sql_operators.asp "SQL Operators")
* [Install MongoDB](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/ "Install MongoDB")

## Tasks

### [0. Create a database](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/0-create_database_if_missing.sql "0. Create a database")

Run with:

```bash
$ cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
```

### [1. First table](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/1-first_table.sql "1. First table")

Run with:

```bash
$ cat 1-first_table.sql | mysql -hlocalhost -uroot -p db_0
```

### [2. List all in table](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/2-list_values.sql "2. List all in table")

Run with:

```bash
$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
```

### [3. First add](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/3-insert_value.sql "3. First add")

Run with:

```bash
$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
```

### [4. Select the best](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/4-best_score.sql "4. Select the best")

Run with:

```bash
$ cat setup.sql
-- Create table and insert data
CREATE TABLE IF NOT EXISTS second_table (
    id INT,
    name VARCHAR(256),
    score INT
);
INSERT INTO second_table (id, name, score) VALUES (1, "Bob", 14);
INSERT INTO second_table (id, name, score) VALUES (2, "Roy", 3);
INSERT INTO second_table (id, name, score) VALUES (3, "John", 10);
INSERT INTO second_table (id, name, score) VALUES (4, "Bryan", 8);

$ cat setup.sql | mysql -hlocalhost -uroot -p db_0
$ cat 4-best_score.sql | mysql -hlocalhost -uroot -p db_0
score   name
14      Bob
10      John
```

### [5. Average](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/5-average.sql "5. Average")

Run with (uses previous setup.sql file):

```bash
$ cat 5-average.sql | mysql -hlocalhost -uroot -p db_0
average
9.25
```

### [6. Temperatures #0](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/6-avg_temperatures.sql "6. Temperatures #0")

Setup and run with:

```bash
$ echo "CREATE DATABASE hbtn_0c_0;" | mysql -hlocalhost -uroot -p
$ curl "https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/272/temperatures.sql" -s | mysql -hlocalhost -uroot -p hbtn_0c_0
$ cat 6-avg_temperatures.sql | mysql -hlocalhost -uroot -p hbtn_0c_0
city    avg_temp
Chandler    72.8627
...
Peoria  66.5392
```

### [7. Temperatures #2](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/7-max_state.sql "7. Temperatures #2")

Use previous commands to create hbtn_0c_0 DB.

Run with:

```bash
$ cat 7-max_state.sql | mysql -hlocalhost -uroot -p hbtn_0c_0
state   max_temp
AZ      110
CA      110
IL      110
```

### [8. Genre ID by show](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/8-genre_id_by_show.sql "8. Genre ID by show")

Setup and run with:

```bash
$ echo "CREATE DATABASE hbtn_0d_tvshows;" | mysql -hlocalhost -uroot -p
$ curl "https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql" -s | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
$ cat 8-genre_id_by_show.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
title   genre_id
Breaking Bad    1
...
The Last Man on Earth   5
```

### [9. No genre](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/9-no_genre.sql "9. No genre")

Use previous commands to create hbtn_0d_tvshows DB.

Run with:

```bash
$ cat 9-no_genre.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
title   genre_id
Better Call Saul        NULL
Homeland        NULL
```

### [10. Number of shows by genre](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/10-count_shows_by_genre.sql "10. Number of shows by genre")

Use previous commands to create hbtn_0d_tvshows DB.

Run with:

```bash
$ cat 10-count_shows_by_genre.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
genre   number_of_shows
Drama   5
...
Fantasy 1
```

### [11. Rotten tomatoes](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/11-rating_shows.sql "11. Rotten tomatoes")

Setup and run with:

```bash
$ echo "CREATE DATABASE hbtn_0d_tvshows_rate;" | mysql -hlocalhost -uroot -p
$ curl "https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows_rate.sql" -s | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
$ cat 11-rating_shows.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
title   rating
Better Call Saul    163
...
New Girl    0
```

### [12. Best genre](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/12-rating_genres.sql "12. Best genre")

Use previous commands to create hbtn_0d_tvshows_rate DB.

Run with:

```bash
$ cat 12-rating_genres.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
name    rating
Drama   150
...
Thriller    40
```

### [13. We are all unique!](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/13-uniq_users.sql "13. We are all unique!")

Run with:

```bash
$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
ERROR 1146 (42S02) at line 1: Table 'holberton.users' doesn\'t exist
$ cat 13-uniq_users.sql | mysql -uroot -p holberton
$ echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");' | mysql -uroot -p holberton
$ echo 'INSERT INTO users (email, name) VALUES ("sylvie@dylan.com", "Sylvie");' | mysql -uroot -p holberton
$ echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Jean");' | mysql -uroot -p holberton
ERROR 1062 (23000) at line 1: Duplicate entry 'bob@dylan.com' for key 'email'
$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
id  email   name
1   bob@dylan.com   Bob
2   sylvie@dylan.com    Sylvie
```

### [14. In and not out](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/14-country_users.sql "14. In and not out")

Run with:

```bash
$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
ERROR 1146 (42S02) at line 1: Table 'holberton.users' doesn
't exist
$ cat 14-country_users.sql | mysql -uroot -p holberton
$ echo 'INSERT INTO users (email, name, country) VALUES ("bob@dylan.com", "Bob", "US");' | mysql -uroot -p holberton
$ echo 'INSERT INTO users (email, name, country) VALUES ("sylvie@dylan.com", "Sylvie", "CO");' | mysql -uroot -p holberton
$ echo 'INSERT INTO users (email, name, country) VALUES ("jean@dylan.com", "Jean", "FR");' | mysql -uroot -p holberton
ERROR 1265 (01000) at line 1: Data truncated for column 'country' at row 1
$ echo 'INSERT INTO users (email, name) VALUES ("john@dylan.com", "John");' | mysql -uroot -p holberton
$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
id  email   name    country
1   bob@dylan.com   Bob US
2   sylvie@dylan.com    Sylvie  CO
3   john@dylan.com  John    US
```

### [15. Best band ever!](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/15-fans.sql "15. Best band ever!")

Run with:

```bash
$ cat metal_bands.sql | mysql -uroot -p holberton
$ cat 15-fans.sql | mysql -uroot -p holberton > tmp_res ; head tmp_res
origin  nb_fans
USA 99349
...
Italy   7178
```

### [16. Old school band](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/16-glam_rock.sql "16. Old school band")

Run with:

```bash
$ cat metal_bands.sql | mysql -uroot -p holberton
$ cat 16-glam_rock.sql | mysql -uroot -p holberton
band_name   lifespan
Alice Cooper    56
...
Hanoi Rocks 0
```

### [17. Buy buy buy](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/17-store.sql "17. Buy buy buy")

Run with:

```bash
$ cat 17-init.sql | mysql -uroot -p holberton
$ cat 17-store.sql | mysql -uroot -p holberton
$ cat 17-main.sql | mysql -uroot -p holberton
name    quantity
apple   10
...
pear    2
```

### [18. Email validation to sent](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/18-valid_email.sql "18. Email validation to sent")

Run with:

```bash
$ cat 18-init.sql | mysql -uroot -p holberton
$ cat 18-valid_email.sql | mysql -uroot -p holberton
$ cat 18-main.sql | mysql -uroot -p holberton
id  email   name    valid_email
1   bob@dylan.com   Bob 0
...
3   jeanne@dylan.com    Jannis  1
```

### [19. Add bonus](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/19-bonus.sql "19. Add bonus")

Run with:

```bash
$ cat 19-init.sql | mysql -uroot -p holberton
$ cat 19-bonus.sql | mysql -uroot -p holberton
$ cat 19-main.sql | mysql -uroot -p holberton
id  name
...
2   4   90
```

### [20. Average score](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/20-average_score.sql "20. Average score")

Run with:

```bash
$ cat 20-init.sql | mysql -uroot -p holberton
$ cat 20-average_score.sql | mysql -uroot -p holberton
$ cat 20-main.sql | mysql -uroot -p holberton
id  name    average_score
...
2   Jeanne  82
```

### [21. Safe divide](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/pipeline/0x00-databases/21-div.sql "21. Safe divide")

Run with:

```bash
$ cat 21-init.sql | mysql -uroot -p holberton
$ cat 21-div.sql | mysql -uroot -p holberton
$ echo "SELECT (a / b) FROM numbers;" | mysql -uroot -p holberton
(a / b)
...
0.75
```

### [22. List all databases]( "22. List all databases")

Run with:

```bash
$ cat 22-list_databases | mongo
```

### [23. Create a database]( "23. Create a database")

Run with:

```bash
$ cat 23-use_or_create_database | mongo
```

### [24. Insert document]("24. Insert document")

Run with:

```bash
$ cat 24-insert | mongo my_db
```

### [25. All documents]( "25. All documents")

Run with:

```bash
$ cat 25-all | mongo my_db
```

### [26. All matches]( "26. All matches")

Run with:

```bash
$ cat 26-match | mongo my_db
```

### [27. Count]( "27. Count")

Run with:

```bash
$ cat 27-count | mongo my_db
```

### [28. Update]( "28. Update")

Run with:

```bash
$ cat 28-update | mongo my_db
```

### [29. Delete by match]( "29. Delete by match")

Run with:

```bash
$ cat 29-delete | mongo my_db
```

### [30. List all documents in Python]( "30. List all documents in Python")

Run with:

```bash
$ ./30-all.py
```

### [31. Insert a document in Python]( "31. Insert a document in Python")

Run with:

```bash
$ ./31-insert_school.py
```

### [32. Change school topics]( "32. Change school topics")

Run with:

```bash
$ ./32-update_topics.py
```

### [33. Where can I learn Python?]( "33. Where can I learn Python?")

Run with:

```bash
$ ./33-schools_by_topic.py
```

### [34. Log stats]( "34. Log stats")

Run with:

```bash
$ curl -o dump.zip -s "https://s3.amazonaws.com/intranet-projects-files/holbertonschool-webstack/411/dump.zip"
$ unzip dump.zip
$ mongorestore dump
$ ./34-log_stats.py
```
