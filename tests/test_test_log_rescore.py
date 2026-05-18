from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_test_log_rescore_honors_disabled_and_min_score_filters_aggregate() -> None:
    response = client.post(
        '/api/test-log/rescore',
        json={
            'test_items': [
                {
                    'test_item': 'ITEM_HIGH',
                    'value': '5',
                    'usl': 10,
                    'lsl': 0,
                    'status': 'PASS',
                },
                {
                    'test_item': 'ITEM_LOW',
                    'value': '0',
                    'usl': 10,
                    'lsl': 0,
                    'status': 'PASS',
                },
                {
                    'test_item': 'ITEM_DISABLED',
                    'value': '5',
                    'usl': 10,
                    'lsl': 0,
                    'status': 'PASS',
                },
            ],
            'scoring_configs': [
                {
                    'test_item_name': 'ITEM_LOW',
                    'scoring_type': 'symmetrical',
                    'enabled': True,
                    'weight': 1,
                    'min_score': 0.8,
                },
                {
                    'test_item_name': 'ITEM_DISABLED',
                    'scoring_type': 'symmetrical',
                    'enabled': False,
                    'weight': 1,
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    score_map = {item['test_item']: item for item in payload['test_item_scores']}

    assert score_map['ITEM_HIGH']['score'] == 10.0
    assert score_map['ITEM_LOW']['score'] == 1.0
    assert score_map['ITEM_DISABLED']['score'] is None
    assert payload['overall_score'] == 10.0
