"""Korean AI-tone advisory pipeline.

Pipeline stages (pure functions passing Pydantic v2 state):

    fetch_feed -> extract_post_text -> segment_korean -> korean_ai_score
        -> build_slack_message -> post_to_slack

The detector reimplements linguistic feature *ideas* from KatFishNet
(ACL 2025, arXiv:2503.00032) from scratch — no code is copied from that
repository (it carries no licence). All output is advisory only.
"""
