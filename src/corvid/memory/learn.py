"""LEARN: give Graphiti the raw email as an episode, constrained by the ontology."""

from datetime import datetime
from email import message_from_string, policy
from email.utils import parsedate_to_datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from corvid.memory.ontology import edge_type_map, edge_types, entity_types


def parse_email(raw: str) -> tuple[str, datetime]:
    """Turn a raw .eml string into (episode body, sent date)."""
    message = message_from_string(raw, policy=policy.default)
    date = parsedate_to_datetime(message["Date"])
    body = (
        f"From: {message['From']}\n"
        f"Subject: {message['Subject']}\n\n"
        f"{message.get_content().strip()}"
    )
    return body, date


async def learn(graphiti: Graphiti, name: str, body: str, date: datetime) -> None:
    """Add one email to memory as an episode."""
    await graphiti.add_episode(
        name=name,
        episode_body=body,
        source_description="customer email",
        reference_time=date,
        source=EpisodeType.text,  # default is message; text fits emails
        entity_types=entity_types,
        edge_types=edge_types,
        edge_type_map=edge_type_map,
        excluded_entity_types=["Entity"],
    )
