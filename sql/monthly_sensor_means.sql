-- Monthly mean per sensor, for the drift analysis.
--
-- A sensor whose monthly mean moves steadily in the same direction as the failure rate
-- will show a strong association inside any window that contains the trend, whether or
-- not it has anything to do with yield. This aggregation is what makes that visible.

SELECT
    m.sensor_id,
    strftime(l.measured_at, '%Y-%m') AS month,
    COUNT(m.value)                   AS n_observed,
    AVG(m.value)                     AS mean_value
FROM measurement m
JOIN lot l USING (record_id)
WHERE m.sensor_id IN {sensor_list}
GROUP BY m.sensor_id, month
ORDER BY m.sensor_id, month;
