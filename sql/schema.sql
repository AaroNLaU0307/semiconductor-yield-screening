-- Warehouse schema for the SECOM process data.
--
-- Fab process data lives in a database, not in flat files, and a yield analyst's work
-- starts with a query rather than with a file read. This schema is the shape that data
-- actually takes in a process warehouse: one row per lot, and one row per measured
-- cell in long form.
--
-- Long form rather than 590 wide columns because that is how sensor data is warehoused,
-- because it makes coverage and missingness natural aggregations rather than contorted
-- column scans, and because adding a sensor becomes an insert rather than a schema
-- migration.
--
-- A NULL `value` means the sensor was not measured for that lot. The full grid is
-- stored, 1567 lots by 590 sensors, so that the long form round trips back to the wide
-- matrix exactly. Omitting unmeasured cells would be more compact and is what a real
-- warehouse often does, but it would make the round trip lossy and this project's whole
-- claim is that no number moved.

CREATE TABLE IF NOT EXISTS lot (
    record_id    INTEGER   NOT NULL,
    measured_at  TIMESTAMP NOT NULL,
    is_fail      BOOLEAN   NOT NULL,
    PRIMARY KEY (record_id)
);

CREATE TABLE IF NOT EXISTS measurement (
    record_id  INTEGER NOT NULL,
    sensor_id  VARCHAR NOT NULL,
    value      DOUBLE,
    PRIMARY KEY (record_id, sensor_id)
);
