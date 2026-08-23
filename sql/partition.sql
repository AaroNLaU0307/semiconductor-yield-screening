-- The chronological train/holdout boundary, expressed as a query rather than a slice.
--
-- DECISIONS.md D1 fixes the split at the first {train_fraction} of records ordered by
-- timestamp. Ordering by (measured_at, record_id) reproduces a stable sort: record_id
-- is the position in the raw file, so lots sharing a timestamp keep their original
-- order and the boundary is deterministic.
--
-- ROW_NUMBER is 1 based, so `position <= floor(n * fraction)` selects exactly the same
-- rows as the Python slice `[:int(floor(n * fraction))]`.

CREATE OR REPLACE VIEW lot_partition AS
WITH ordered AS (
    SELECT
        record_id,
        measured_at,
        is_fail,
        ROW_NUMBER() OVER (ORDER BY measured_at, record_id) AS position,
        COUNT(*)     OVER ()                                AS n_total
    FROM lot
)
SELECT
    record_id,
    measured_at,
    is_fail,
    position,
    CASE
        WHEN position <= FLOOR(n_total * {train_fraction}) THEN 'train'
        ELSE 'holdout'
    END AS partition
FROM ordered;
