# SQL SELECT DISTINCT Statement

The `SELECT DISTINCT` statement is used to return only distinct (different) values.

Inside a table, a column often contains many duplicate values; and sometimes you only want to list the different (distinct) values.

```sql
SELECT DISTINCT column1, column2, ...
FROM table_name;
```

The following SQL statement selects only the DISTINCT values from the "Country" column in the "Customers" table:

```sql
SELECT DISTINCT Country FROM Customers;
```

The following SQL statement lists the number of different (distinct) customer countries:

```sql
SELECT COUNT(DISTINCT Country) FROM Customers;
```
