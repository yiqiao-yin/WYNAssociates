# SQL Server CAST() Function

The `CAST()` function converts a value (of any type) into a specified datatype.

Tip: Also look at the `CONVERT()` function.

```sql
CAST(expression AS datatype(length))
```

Example:

```sql
SELECT CAST(25.65 AS varchar);
```

```sql
SELECT CAST('2017-08-25' AS datetime);
```
