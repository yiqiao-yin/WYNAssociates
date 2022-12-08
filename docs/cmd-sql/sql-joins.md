# Different Types of SQL JOINs

Here are the different types of the JOINs in SQL:

- `(INNER) JOIN`: Returns records that have matching values in both tables
- `LEFT (OUTER) JOIN`: Returns all records from the left table, and the matched records from the right table
- `RIGHT (OUTER) JOIN`: Returns all records from the right table, and the matched records from the left table
- `FULL (OUTER) JOIN`: Returns all records when there is a match in either left or right table

![image](https://www.w3schools.com/sql/img_innerjoin.gif) ![image](https://www.w3schools.com/sql/img_leftjoin.gif) ![image](https://www.w3schools.com/sql/img_rightjoin.gif) ![image](https://www.w3schools.com/sql/img_fulljoin.gif)


## SQL INNER JOIN Keyword

The `INNER JOIN` keyword selects records that have matching values in both tables.

```sql
SELECT column_name(s)
FROM table1
INNER JOIN table2
ON table1.column_name = table2.column_name;
```

![image](https://www.w3schools.com/sql/img_innerjoin.gif)

## SQL LEFT JOIN Keyword

The `LEFT JOIN` keyword returns all records from the left table (table1), and the matching records from the right table (table2). The result is 0 records from the right side, if there is no match.

```sql
SELECT column_name(s)
FROM table1
LEFT JOIN table2
ON table1.column_name = table2.column_name;
```

![image](https://www.w3schools.com/sql/img_leftjoin.gif)

## SQL RIGHT JOIN Keyword

The `RIGHT JOIN` keyword returns all records from the right table (table2), and the matching records from the left table (table1). The result is 0 records from the left side, if there is no match.

```sql
SELECT column_name(s)
FROM table1
RIGHT JOIN table2
ON table1.column_name = table2.column_name;
```

![image](https://www.w3schools.com/sql/img_rightjoin.gif)

Note: In some databases `RIGHT JOIN` is called `RIGHT OUTER JOIN`.

## SQL FULL OUTER JOIN Keyword

The `FULL OUTER JOIN` keyword returns all records when there is a match in left (table1) or right (table2) table records.

```sql
SELECT column_name(s)
FROM table1
FULL OUTER JOIN table2
ON table1.column_name = table2.column_name
WHERE condition;
```

![image](https://www.w3schools.com/sql/img_fulljoin.gif)

Note: `FULL OUTER JOIN` can potentially return very large result-sets!

