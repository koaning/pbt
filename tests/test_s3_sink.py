"""Tests for S3Sink with mocked boto3."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest


class MockClientError(Exception):
    """Mock boto3 ClientError."""

    def __init__(self, error_response, operation_name):
        self.response = error_response


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for S3 operations."""
    mock_boto3 = MagicMock()
    client = MagicMock()
    mock_boto3.client.return_value = client
    client.exceptions.ClientError = MockClientError

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        yield client, mock_boto3.client


@pytest.fixture
def s3_sink(mock_boto3_client):
    """Create S3Sink with mocked boto3."""
    from pbt.sinks.s3 import S3Sink

    return S3Sink(
        bucket="test-bucket",
        prefix="warehouse",
        storage_options={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
        },
    )


def test_init_stores_config(s3_sink):
    assert s3_sink.bucket == "test-bucket"
    assert s3_sink.prefix == "warehouse"
    assert s3_sink.storage_options["aws_access_key_id"] == "test-key"


def test_init_strips_slashes_from_prefix(mock_boto3_client):
    from pbt.sinks.s3 import S3Sink

    sink = S3Sink(bucket="b", prefix="/foo/bar/")
    assert sink.prefix == "foo/bar"


def test_init_empty_prefix(mock_boto3_client):
    from pbt.sinks.s3 import S3Sink

    sink = S3Sink(bucket="b", prefix="")
    assert sink.prefix == ""


def test_init_passes_endpoint_url_to_boto3(mock_boto3_client):
    from pbt.sinks.s3 import S3Sink

    client, client_factory = mock_boto3_client

    S3Sink(
        bucket="b",
        storage_options={"endpoint_url": "http://minio:9000"},
    )

    call_kwargs = client_factory.call_args[1]
    assert call_kwargs.get("endpoint_url") == "http://minio:9000"


def test_s3_uri_simple(s3_sink):
    assert (
        s3_sink._s3_uri("table.parquet") == "s3://test-bucket/warehouse/table.parquet"
    )


def test_s3_uri_with_parts(s3_sink):
    uri = s3_sink._s3_uri("events", "date=2025-01-05", "data.parquet")
    assert uri == "s3://test-bucket/warehouse/events/date=2025-01-05/data.parquet"


def test_s3_uri_no_prefix(mock_boto3_client):
    from pbt.sinks.s3 import S3Sink

    sink = S3Sink(bucket="b", prefix="")
    assert sink._s3_uri("table.parquet") == "s3://b/table.parquet"


def test_s3_key_simple(s3_sink):
    assert s3_sink._s3_key("table.parquet") == "warehouse/table.parquet"


def test_s3_key_with_parts(s3_sink):
    key = s3_sink._s3_key("events", "date=2025-01-05", "data.parquet")
    assert key == "warehouse/events/date=2025-01-05/data.parquet"


def test_exists_single_file(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    client.head_object.return_value = {}
    assert s3_sink.exists("my_table") is True
    client.head_object.assert_called_with(
        Bucket="test-bucket", Key="warehouse/my_table.parquet"
    )


def test_exists_partitioned(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    error = MockClientError({"Error": {"Code": "404"}}, "HeadObject")
    client.head_object.side_effect = error
    client.list_objects_v2.return_value = {"KeyCount": 1}

    assert s3_sink.exists("events") is True
    client.list_objects_v2.assert_called_with(
        Bucket="test-bucket",
        Prefix="warehouse/events/",
        MaxKeys=1,
    )


def test_not_exists(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    error = MockClientError({"Error": {"Code": "404"}}, "HeadObject")
    client.head_object.side_effect = error
    client.list_objects_v2.return_value = {"KeyCount": 0}

    assert s3_sink.exists("missing") is False


def test_list_partitions_single_level(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "warehouse/events/date=2025-01-01/data.parquet"},
                {"Key": "warehouse/events/date=2025-01-02/data.parquet"},
                {"Key": "warehouse/events/date=2025-01-03/data.parquet"},
            ]
        }
    ]

    partitions = s3_sink.list_partitions("events")
    assert partitions == ["date=2025-01-01", "date=2025-01-02", "date=2025-01-03"]


def test_list_partitions_multi_level(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "warehouse/events/date=2025-01-01/user=alice/data.parquet"},
                {"Key": "warehouse/events/date=2025-01-01/user=bob/data.parquet"},
                {"Key": "warehouse/events/date=2025-01-02/user=alice/data.parquet"},
            ]
        }
    ]

    partitions = s3_sink.list_partitions("events")
    assert sorted(partitions) == [
        "date=2025-01-01/user=alice",
        "date=2025-01-01/user=bob",
        "date=2025-01-02/user=alice",
    ]


def test_list_partitions_empty(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{"Contents": []}]

    partitions = s3_sink.list_partitions("events")
    assert partitions == []


def test_delete_single_partition(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "warehouse/events/date=2025-01-01/data.parquet"},
            ]
        }
    ]

    s3_sink.delete_partitions("events", ["date=2025-01-01"])

    client.delete_objects.assert_called_once_with(
        Bucket="test-bucket",
        Delete={"Objects": [{"Key": "warehouse/events/date=2025-01-01/data.parquet"}]},
    )


def test_delete_batches_over_1000(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator

    objects = [
        {"Key": f"warehouse/events/date=2025-01-01/file{i}.parquet"}
        for i in range(1500)
    ]
    paginator.paginate.return_value = [{"Contents": objects}]

    s3_sink.delete_partitions("events", ["date=2025-01-01"])

    assert client.delete_objects.call_count == 2


def test_write_single_dry_run(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    client.head_object.side_effect = MockClientError(
        {"Error": {"Code": "404"}}, "HeadObject"
    )

    df = pl.DataFrame({"a": [1, 2, 3]})
    result = s3_sink.write(df, "test_table", dry_run=True)

    assert result.table_name == "test_table"
    assert result.rows_to_write == 3
    assert result.destination == "s3://test-bucket/warehouse/test_table.parquet"


def test_write_partitioned_dry_run(s3_sink, mock_boto3_client):
    client, _ = mock_boto3_client
    client.head_object.side_effect = MockClientError(
        {"Error": {"Code": "404"}}, "HeadObject"
    )

    df = pl.DataFrame(
        {
            "value": [1, 2, 3],
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
        }
    )
    result = s3_sink.write(df, "events", partition_by=["date"], dry_run=True)

    assert result.table_name == "events"
    assert result.rows_to_write == 3
    assert set(result.partitions_affected) == {"date=2025-01-01", "date=2025-01-02"}
    assert result.partition_operations["date=2025-01-01"] == "create"
    assert result.partition_operations["date=2025-01-02"] == "create"


def test_write_missing_partition_column_raises(s3_sink):
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Partition columns not in DataFrame"):
        s3_sink.write(df, "events", partition_by=["missing_col"])
