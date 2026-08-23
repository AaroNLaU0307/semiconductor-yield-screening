-- Unsupervised structure discovery over the missingness matrix.
--
-- The sparse band sensors are not measured on every lot. Grouping them by their exact
-- missing/present vector asks whether those absences are independent per sensor or
-- whether sensors are measured together in blocks. They are measured in blocks: 52
-- sensors collapse to a handful of distinct patterns, which is the signature of
-- metrology sampling plans rather than of sensors failing at random.
--
-- Exact grouping rather than a clustering algorithm, because the patterns are exact.
-- Running K-Means here would impose a distance metric on data that needs none and
-- would turn an exact answer into an approximate one.

WITH sparse_sensors AS (
    SELECT sensor_id FROM ({drop_rules}) WHERE is_sparse_band
),
patterns AS (
    SELECT
        m.sensor_id,
        string_agg(CASE WHEN m.value IS NULL THEN '1' ELSE '0' END, '' ORDER BY p.position)
            AS missing_pattern,
        (COUNT(*) - COUNT(m.value))::DOUBLE / COUNT(*) AS train_missing_rate
    FROM measurement m
    JOIN lot_partition p USING (record_id)
    JOIN sparse_sensors s ON s.sensor_id = m.sensor_id
    WHERE p.partition = 'train'
    GROUP BY m.sensor_id
)
SELECT
    MIN(sensor_id)                      AS representative,
    COUNT(*)                            AS n_sensors,
    list(sensor_id ORDER BY sensor_id)  AS members,
    MIN(train_missing_rate)             AS train_missing_rate,
    1.0 - MIN(train_missing_rate)       AS train_coverage
FROM patterns
GROUP BY missing_pattern
ORDER BY representative;
