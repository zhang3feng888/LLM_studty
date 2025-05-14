curl -X POST "http://127.0.0.1:8000/items" \
    -d '{
        "name": "Foo",
        "description": "The pretender",
        "price": 1.2, 
        "tax": 10
    }'