"""Populate the local SQLite database with rich dummy data for development."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.database import Base, SessionLocal, engine
from sqlalchemy import insert

from app.models import (
    Candidate,
    CommunityComment,
    CommunityPost,
    CommunityPostLike,
    Data,
    Dataset,
    Denoise,
    Lightcurve,
    PipelineStep,
    PinnedProject,
    Project,
    ProjectRepository,
    ProjectUser,
    Repository,
    Session as PipelineSession,
    StarredRepository,
    User,
    UserProfile,
    dataset_data_association,
)
from app.security import hash_password


UTC = timezone.utc


def _paragraphs_to_html(content: str) -> str:
    paragraphs = [line.strip() for line in content.split("\n") if line.strip()]
    if not paragraphs:
        return "<p></p>"
    return "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


async def reset_schema(drop_existing: bool) -> None:
    """Drop and recreate tables to start from a clean slate."""

    async with engine.begin() as conn:
        if drop_existing:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def seed_dummy_data() -> None:
    """Insert interconnected dummy records covering key workflows."""

    base_time = datetime.now(UTC) - timedelta(days=42)

    async with SessionLocal() as session:
        # --- Users and profiles ---
        user_specs = [
            {
                "key": "aria",
                "email": "aria.fernandez@nups.dev",
                "bio": "Planet hunter cross-matching Kepler lightcurves with new models.",
                "avatar": "https://example.com/avatars/aria.png",
                "password": "aria123!",
            },
            {
                "key": "bianca",
                "email": "bianca.cho@nups.dev",
                "bio": "Data engineer automating transient ingestion for observatories in Seoul.",
                "avatar": "https://example.com/avatars/bianca.png",
                "password": "bianca123!",
            },
            {
                "key": "casper",
                "email": "casper.nguyen@nups.dev",
                "bio": "Citizen scientist curating lunar eclipse uploads for the weekend community.",
                "avatar": "https://example.com/avatars/casper.png",
                "password": "casper123!",
            },
            {
                "key": "dimitri",
                "email": "dimitri.ivanov@nups.dev",
                "bio": "Observatory manager benchmarking denoise pipelines across hemispheres.",
                "avatar": "https://example.com/avatars/dimitri.png",
                "password": "dimitri123!",
            },
        ]

        users: dict[str, User] = {}
        for spec in user_specs:
            user = User(
                email=spec["email"],
                password_hash=hash_password(spec["password"]),
            )
            user.profile = UserProfile(bio=spec["bio"], avatar_url=spec["avatar"])
            session.add(user)
            users[spec["key"]] = user

        await session.flush()

        # --- Projects ---
        project_specs = [
            {
                "key": "kepler",
                "name": "Kepler Reprocessing",
                "description": "Re-run legacy Kepler targets with modern detrending.",
                "start_offset": 120,
            },
            {
                "key": "transient",
                "name": "Transient Watch",
                "description": "Community-powered transient detection and triage.",
                "start_offset": 75,
            },
            {
                "key": "lunar",
                "name": "Lunar Lab",
                "description": "Aggregated lunar eclipse captures for teaching labs.",
                "start_offset": 35,
            },
            {
                "key": "aurora",
                "name": "Aurora Atlas",
                "description": "Coordinated aurora imaging across polar research stations.",
                "start_offset": 52,
            },
        ]

        projects: dict[str, Project] = {}
        for spec in project_specs:
            project = Project(
                name=spec["name"],
                description=spec["description"],
                start_date=base_time - timedelta(days=spec["start_offset"]),
            )
            session.add(project)
            projects[spec["key"]] = project

        await session.flush()

        # --- Repositories per user ---
        repo_specs = [
            {
                "owner": "aria",
                "repos": [
                    {
                        "name": "kepler-quarter-14",
                        "description": "Quarter 14 raw flux exports for hot Jupiter sample.",
                    },
                    {
                        "name": "kepler-threshold-crossing",
                        "description": "Threshold crossing events filtered for triage.",
                    },
                ],
            },
            {
                "owner": "bianca",
                "repos": [
                    {
                        "name": "transient-ingestion-pipeline",
                        "description": "Streaming ingestion with adaptive sky partitioning.",
                    },
                    {
                        "name": "spectral-baselines",
                        "description": "Reference spectra bundles for rapid classification.",
                    },
                    {
                        "name": "infra-metrics",
                        "description": "Airflow metrics snapshots used for observability.",
                    },
                ],
            },
            {
                "owner": "casper",
                "repos": [
                    {
                        "name": "lunar-lights",
                        "description": "Citizen uploads from the March lunar eclipse.",
                    },
                    {
                        "name": "school-outreach",
                        "description": "Classroom-ready subsets with narration scripts.",
                    },
                ],
            },
            {
                "owner": "dimitri",
                "repos": [
                    {
                        "name": "denoise-benchmarks",
                        "description": "FFTs and denoise comparisons for southern hemisphere sites.",
                    },
                ],
            },
        ]

        repositories: list[Repository] = []
        for spec in repo_specs:
            owner = users[spec["owner"]]
            for repo_data in spec["repos"]:
                repo = Repository(
                    owner=owner,
                    name=repo_data["name"],
                    description=repo_data["description"],
                )
                session.add(repo)
                repositories.append(repo)

        await session.flush()

        # --- Datasets, data files, sessions, and pipeline steps ---
        statuses = ["completed", "running", "failed", "queued"]
        pipeline_templates = [
            [
                {"name": "ingest", "status": "completed"},
                {"name": "calibration", "status": "completed"},
                {"name": "lightcurve", "status": "completed"},
            ],
            [
                {"name": "ingest", "status": "completed"},
                {"name": "registration", "status": "running"},
                {"name": "classification", "status": "pending"},
            ],
            [
                {"name": "ingest", "status": "completed"},
                {"name": "denoise", "status": "failed"},
                {"name": "fallback", "status": "queued"},
            ],
        ]

        for index, repo in enumerate(repositories, start=1):
            for version in range(1, 3):
                dataset = Dataset(
                    repository=repo,
                    version=version,
                    captured_at=base_time + timedelta(days=index * version),
                )
                session.add(dataset)
                await session.flush()
                for data_idx in range(2):
                    data_hash = uuid.uuid4().hex
                    data_item = Data(
                        hash=f"{repo.name}-{version}-{data_idx}-{data_hash}",
                        fits_original_path=f"/data/{repo.name}/raw_v{version}_{data_idx}.fits",
                        fits_image_path=f"/data/{repo.name}/preview_v{version}_{data_idx}.png",
                        fits_data_json={
                            "exposure_seconds": 30 + (data_idx * 10),
                            "filter": random.choice(["g", "r", "i", "z"]),
                        },
                        metadata_json={
                            "observer": repo.owner.email,
                            "notes": "Synthetic dummy capture for local testing.",
                        },
                    )
                    session.add(data_item)
                    await session.flush()
                    await session.execute(
                        insert(dataset_data_association).values(
                            dataset_id=dataset.id,
                            data_id=data_item.id,
                        )
                    )

                status = statuses[(index + version) % len(statuses)]
                start_time = base_time + timedelta(days=index)
                finish_time = (
                    start_time + timedelta(minutes=60)
                    if status == "completed"
                    else None
                )
                run_id = uuid.uuid4()
                template = pipeline_templates[(index + version) % len(pipeline_templates)]
                pipeline_session = PipelineSession(
                    run_id=run_id,
                    repository=repo,
                    dataset=dataset,
                    data_version=dataset.version,
                    current_step=template[-1]["name"],
                    status=status,
                    progress=85 if status == "completed" else 40 if status == "running" else 10,
                    started_at=start_time,
                    finished_at=finish_time,
                )
                session.add(pipeline_session)

                # Pipeline steps mirror the templates but adapt status variations.
                for order, step in enumerate(template, start=1):
                    step_status = step["status"]
                    if status == "failed" and step["name"] == "denoise":
                        step_status = "failed"
                    pipeline_step = PipelineStep(
                        run_id=run_id,
                        step_name=step["name"],
                        status=step_status,
                        progress=min(100, order * 40),
                        data={"order": order},
                        log=f"{step['name']} step auto-generated for seed data.",
                        started_at=start_time + timedelta(minutes=order * 5),
                        finished_at=(
                            start_time + timedelta(minutes=order * 12)
                            if step_status in {"completed", "failed"}
                            else None
                        ),
                    )
                    session.add(pipeline_step)

                # Attach optional analysis records for completed runs.
                if status == "completed":
                    session.add(
                        Lightcurve(
                            session=pipeline_session,
                            data={"points": [random.random() for _ in range(5)]},
                        )
                    )
                    session.add(
                        Denoise(
                            session=pipeline_session,
                            data={"smoothing_factor": 0.42},
                        )
                    )
                    session.add(
                        Candidate(
                            session=pipeline_session,
                            data={"score": 0.91},
                            is_verified=True,
                        )
                    )
                elif status == "failed":
                    session.add(
                        Candidate(
                            session=pipeline_session,
                            data={"score": 0.12},
                            is_verified=False,
                        )
                    )

        await session.flush()

        # --- Project memberships ---
        memberships = [
            ("kepler", "aria", "lead"),
            ("kepler", "bianca", "data"),
            ("transient", "bianca", "maintainer"),
            ("transient", "casper", "reviewer"),
            ("transient", "dimitri", "observer"),
            ("lunar", "casper", "curator"),
            ("lunar", "aria", "mentor"),
            ("lunar", "dimitri", "instructor"),
            ("aurora", "dimitri", "coordinator"),
            ("aurora", "bianca", "automation"),
        ]
        for project_key, user_key, role in memberships:
            session.add(
                ProjectUser(
                    project=projects[project_key],
                    user=users[user_key],
                    role=role,
                    joined_at=base_time + timedelta(days=len(role)),
                )
            )

        # --- Repository uploads mapped into projects (>= 10 records) ---
        upload_pairs = [
            ("kepler", "kepler-quarter-14", "aria"),
            ("kepler", "kepler-threshold-crossing", "bianca"),
            ("kepler", "spectral-baselines", "aria"),
            ("kepler", "infra-metrics", "bianca"),
            ("transient", "transient-ingestion-pipeline", "bianca"),
            ("transient", "spectral-baselines", "dimitri"),
            ("transient", "infra-metrics", "bianca"),
            ("transient", "denoise-benchmarks", "dimitri"),
            ("lunar", "lunar-lights", "casper"),
            ("lunar", "school-outreach", "aria"),
            ("lunar", "kepler-threshold-crossing", "casper"),
            ("lunar", "denoise-benchmarks", "dimitri"),
            ("aurora", "denoise-benchmarks", "dimitri"),
            ("aurora", "spectral-baselines", "bianca"),
        ]

        repo_lookup = {repo.name: repo for repo in repositories}
        for project_key, repo_name, uploader_key in upload_pairs:
            session.add(
                ProjectRepository(
                    project=projects[project_key],
                    repository=repo_lookup[repo_name],
                    uploader=users[uploader_key],
                    uploaded_at=base_time + timedelta(hours=len(repo_name)),
                )
            )

        # --- Favorites / stars and pins ---
        starred_map = {
            "aria": "spectral-baselines",
            "bianca": "denoise-benchmarks",
            "casper": "school-outreach",
            "dimitri": "kepler-quarter-14",
        }
        for user_key, repo_name in starred_map.items():
            session.add(
                StarredRepository(
                    user=users[user_key],
                    repository=repo_lookup[repo_name],
                    starred_at=base_time + timedelta(days=len(repo_name)),
                )
            )

        pinned_map = {
            "aria": "kepler",
            "bianca": "transient",
            "casper": "lunar",
            "dimitri": "aurora",
        }
        for user_key, project_key in pinned_map.items():
            session.add(
                PinnedProject(
                    user=users[user_key],
                    project=projects[project_key],
                    pinned_at=base_time + timedelta(days=len(project_key)),
                    position=list(pinned_map.keys()).index(user_key) + 1,
                )
            )

        # --- Community posts, comments, and likes ---
        community_specs = [
            {
                "title": "Aurora Watch: Solar Storm Update for Tonight",
                "content": (
                    "NOAA just bumped tonight's aurora forecast to G3 levels. If you're in mid-latitudes, "
                    "swing your all-sky rigs north and watch the Kp charts after 22:00 UTC. Drop your captures "
                    "in the gallery thread so we can stitch a global montage."
                ),
                "category": "announcements",
                "author": "dimitri",
                "project": "aurora",
                "days_after": 35,
                "likes": ["aria", "bianca", "casper"],
                "comments": [
                    {
                        "author": "aria",
                        "hours_after": 1.5,
                        "content": "Thanks for the flare warning—I'll repoint the Kepler rigs for a few hours.",
                    },
                    {
                        "author": "casper",
                        "hours_after": 3.2,
                        "content": "Cloud deck over Busan is thinning, hoping to grab at least a timelapse streak!",
                    },
                ],
            },
            {
                "title": "Astrophoto Gallery: NGC 1300 Narrowband Composite",
                "content": (
                    "Stacked four clear nights of data from the Luna Lab queue using dual-narrowband filters. "
                    "Sharing both the straight SHO and the blended HOO treatment for anyone teaching colour mapping."
                ),
                "category": "astrophoto-gallery",
                "author": "casper",
                "project": None,
                "days_after": 34,
                "likes": ["aria", "dimitri"],
                "comments": [
                    {
                        "author": "bianca",
                        "hours_after": 2.0,
                        "content": "Would love to see the PixInsight flow you used—those dust lanes are razor sharp!",
                    }
                ],
            },
            {
                "title": "Project HaloSpectra v2.1 Released",
                "content": (
                    "Batch inference now runs 30% faster with the new GPU profile presets. We also wired in the "
                    "Mission Control vetting checklist so flagged candidates sync straight to review boards."
                ),
                "category": "project-showcase",
                "author": "bianca",
                "project": "kepler",
                "days_after": 33,
                "likes": ["aria", "dimitri"],
                "comments": [
                    {
                        "author": "dimitri",
                        "hours_after": 4.5,
                        "content": "Great work—does this include the new dark-frame bootstrap we tested last week?",
                    }
                ],
            },
            {
                "title": "Upload Hall of Fame: SpectraSuite Crosses 15k Downloads",
                "content": (
                    "Shout-out to everyone contributing spectral baselines—SpectraSuite passed 15k downloads this "
                    "morning. Share your favourite automations so we can feature them in the onboarding docs."
                ),
                "category": "upload-hall-of-fame",
                "author": "aria",
                "project": "transient",
                "days_after": 32,
                "likes": ["bianca", "dimitri"],
                "comments": [],
            },
            {
                "title": "Mission Briefing: October Deep Field Campaign",
                "content": (
                    "Kicking off a month-long cadence focused on M31 satellite candidates. We'll sync on Thursday to "
                    "finalise exposure presets and decide which observatories take the late-night windows."
                ),
                "category": "announcements",
                "author": "aria",
                "project": None,
                "days_after": 31,
                "likes": ["bianca", "casper"],
                "comments": [],
            },
            {
                "title": "Lunar Occultation Tracker Beta Seeking Observers",
                "content": (
                    "We just pushed the beta predictor for lunar occultations of bright radio sources. Looking for "
                    "Asia-Pacific observers to validate tomorrow night's ephemerides—DM if you can cover 12-16 UTC."
                ),
                "category": "project-showcase",
                "author": "casper",
                "project": "lunar",
                "days_after": 30,
                "likes": ["aria"],
                "comments": [
                    {
                        "author": "dimitri",
                        "hours_after": 2.3,
                        "content": "Can route the Santiago dish for a cross-check if you share the tracking script.",
                    }
                ],
            },
        ]

        for spec in community_specs:
            created_at = base_time + timedelta(days=spec["days_after"])
            content_html = _paragraphs_to_html(spec["content"])
            post = CommunityPost(
                title=spec["title"],
                content=content_html,
                category=spec["category"],
                author=users[spec["author"]],
                linked_project=projects.get(spec["project"]) if spec["project"] else None,
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(post)
            await session.flush()

            for comment_spec in spec["comments"]:
                comment_created_at = created_at + timedelta(hours=comment_spec.get("hours_after", 2))
                session.add(
                    CommunityComment(
                        post=post,
                        author=users[comment_spec["author"]],
                        content=comment_spec["content"],
                        created_at=comment_created_at,
                    )
                )

            for like_key in spec["likes"]:
                session.add(
                    CommunityPostLike(
                        post_id=post.id,
                        user_id=users[like_key].id,
                        liked_at=created_at + timedelta(hours=1),
                    )
                )

        await session.commit()

    # SQLite keeps connections open; dispose for clean exit.
    await engine.dispose()


async def main(drop_existing: bool) -> None:
    await reset_schema(drop_existing)
    await seed_dummy_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the database with dummy data.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Preserve existing tables (skip DROP ALL) before seeding.",
    )
    args = parser.parse_args()
    asyncio.run(main(drop_existing=not args.keep_existing))
