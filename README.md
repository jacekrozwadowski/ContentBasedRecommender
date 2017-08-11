# ContentBasedRecommender
Simple Content Based recommendation engine

## Try it out!

Training parameters:
```
--redis_url=redis://<host>:<port>/<database no>
--training 
--use-hashing --lsa=10
```

Training parameters:
```
--redis_url=redis://<host>:<port>/<database>
--item_id=<item no>
--n-similar=<no of similar items>
```