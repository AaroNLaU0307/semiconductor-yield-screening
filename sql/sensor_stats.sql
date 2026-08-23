-- Per sensor extraction primitive over a chosen partition.
--
-- `{partition_predicate}` selects the scope: `p.partition = 'train'` for everything the
-- declared drop rules need, or `TRUE` to describe the dataset as landed. Fitting any
-- transform requires the train scope, and the drop rule caller always passes it, but
-- the data quality report legitimately describes the whole file too.
--
-- Everything the declared drop rules need, in one pass: coverage, distinct value
-- count, the share held by the most common value, and a fingerprint of the full value
-- vector including its missing positions.
--
-- The fingerprint is what makes duplicate detection a grouping rather than a pairwise
-- comparison of 590 columns. Casting a DOUBLE to VARCHAR in DuckDB uses the shortest
-- round trip representation, which is injective, so two sensors share a fingerprint if
-- and only if their value vectors are identical.

WITH train_cells AS (
    SELECT m.sensor_id, m.value, p.position
    FROM measurement m
    JOIN lot_partition p USING (record_id)
    WHERE {partition_predicate}
),
value_counts AS (
    SELECT sensor_id, value, COUNT(*) AS n
    FROM train_cells
    WHERE value IS NOT NULL
    GROUP BY sensor_id, value
),
modal AS (
    SELECT
        sensor_id,
        SUM(n)   AS n_non_null,
        COUNT(*) AS n_distinct_values,
        MAX(n)   AS modal_count
    FROM value_counts
    GROUP BY sensor_id
),
fingerprints AS (
    SELECT
        sensor_id,
        MD5(string_agg(COALESCE(CAST(value AS VARCHAR), '<NA>'), '|' ORDER BY position))
            AS fingerprint
    FROM train_cells
    GROUP BY sensor_id
),
totals AS (
    SELECT sensor_id, COUNT(*) AS n_records
    FROM train_cells
    GROUP BY sensor_id
)
SELECT
    t.sensor_id,
    t.n_records,
    COALESCE(m.n_non_null, 0)                                   AS n_observed,
    t.n_records - COALESCE(m.n_non_null, 0)                     AS n_missing,
    (t.n_records - COALESCE(m.n_non_null, 0))::DOUBLE / t.n_records AS missing_rate,
    COALESCE(m.n_non_null, 0)::DOUBLE / t.n_records             AS coverage,
    COALESCE(m.n_distinct_values, 0)                            AS n_distinct_values,
    CASE WHEN m.n_non_null > 0
         THEN m.modal_count::DOUBLE / m.n_non_null
    END                                                          AS modal_share,
    f.fingerprint
FROM totals t
LEFT JOIN modal m USING (sensor_id)
LEFT JOIN fingerprints f USING (sensor_id)
ORDER BY t.sensor_id;
