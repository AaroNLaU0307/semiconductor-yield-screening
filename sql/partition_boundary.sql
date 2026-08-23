-- The boundary itself, for reporting: sizes, failure counts, and the two timestamps
-- either side of the split.

SELECT
    partition,
    COUNT(*)                                  AS n_records,
    SUM(CASE WHEN is_fail THEN 1 ELSE 0 END)  AS n_fail,
    AVG(CASE WHEN is_fail THEN 1.0 ELSE 0.0 END) AS fail_rate,
    MIN(measured_at)                          AS first_measured_at,
    MAX(measured_at)                          AS last_measured_at
FROM lot_partition
GROUP BY partition
ORDER BY MIN(position);
