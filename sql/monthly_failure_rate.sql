-- Failure rate by calendar month, across the whole record.
--
-- This is the measurement that drives DECISIONS.md D0 and the drift finding: a process
-- whose failure rate moves by a factor of seven across three months is not exchangeable
-- across time, so any procedure treating the records as an unordered bag measures the
-- wrong thing.

SELECT
    strftime(measured_at, '%Y-%m')               AS month,
    COUNT(*)                                     AS n_records,
    SUM(CASE WHEN is_fail THEN 1 ELSE 0 END)     AS n_fail,
    AVG(CASE WHEN is_fail THEN 1.0 ELSE 0.0 END) AS fail_rate
FROM lot
GROUP BY month
ORDER BY month;
