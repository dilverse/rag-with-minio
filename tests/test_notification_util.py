import unittest

from utils.notification_util import partition_records


class PartitionRecordsTest(unittest.TestCase):
    def test_uses_record_level_event_names(self):
        payload = {
            "Records": [
                {
                    "eventName": "s3:ObjectCreated:Put",
                    "s3": {"bucket": {"name": "custom-corpus"}, "object": {"key": "doc.pdf"}},
                },
                {
                    "eventName": "s3:ObjectRemoved:Delete",
                    "s3": {"bucket": {"name": "custom-corpus"}, "object": {"key": "doc.pdf"}},
                },
            ]
        }

        created_records, removed_records = partition_records(payload)

        self.assertEqual(len(created_records), 1)
        self.assertEqual(len(removed_records), 1)

    def test_accepts_multipart_upload_events(self):
        payload = {
            "Records": [
                {
                    "eventName": "s3:ObjectCreated:CompleteMultipartUpload",
                    "s3": {"bucket": {"name": "custom-corpus"}, "object": {"key": "catalog.pdf"}},
                }
            ]
        }

        created_records, removed_records = partition_records(payload)

        self.assertEqual(len(created_records), 1)
        self.assertEqual(len(removed_records), 0)

    def test_falls_back_to_top_level_event_name(self):
        payload = {
            "EventName": "s3:ObjectCreated:Put",
            "Records": [
                {
                    "s3": {"bucket": {"name": "warehouse"}, "object": {"key": "metadata/doc.json"}},
                }
            ],
        }

        created_records, removed_records = partition_records(payload)

        self.assertEqual(len(created_records), 1)
        self.assertEqual(len(removed_records), 0)


if __name__ == "__main__":
    unittest.main()
