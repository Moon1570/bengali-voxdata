"""Flask web UI for seeding speaker POIs (Persons of Interest).

Researchers add a POI with metadata + a reference image, browse all POIs, set a
pipeline status (label), and attach free-form tags. Data is written through
``webui.db`` into a SQLite database compatible with the pipeline manifest.

Run locally:
    python -m webui.app          # http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import re
import uuid

from flask import (
    Flask, abort, flash, redirect, render_template, request,
    send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

from . import db

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")


def create_app() -> Flask:
    """Application factory: configure, ensure the DB exists, register routes."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("POI_SECRET", "dev-poi-secret")
    app.config["UPLOAD_DIR"] = UPLOAD_DIR
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB image cap
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    db.init_db()
    _register_routes(app)
    return app


def _slugify(name: str) -> str:
    """Build a filesystem/id-safe slug from a display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "poi"


def _save_image(file_storage, speaker_id: str) -> str | None:
    """Persist an uploaded image under uploads/<speaker_id>/ and return its
    path relative to the upload dir, or ``None`` if no/invalid file given."""
    if file_storage is None or not file_storage.filename:
        return None
    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise ValueError(f"unsupported image type {ext!r}")
    seed_dir = os.path.join(UPLOAD_DIR, speaker_id)
    os.makedirs(seed_dir, exist_ok=True)
    fname = secure_filename(f"{uuid.uuid4().hex}{ext}")
    file_storage.save(os.path.join(seed_dir, fname))
    return os.path.join(speaker_id, fname)


def _register_routes(app: Flask) -> None:
    """Attach all view functions to the app."""

    @app.route("/")
    def index():
        """List every POI with thumbnail, status, and tags."""
        return render_template("list.html", speakers=db.list_speakers(),
                               statuses=db.STATUSES)

    @app.route("/poi/new", methods=["GET", "POST"])
    def new_poi():
        """Render the add-POI form (GET) and create the speaker (POST)."""
        if request.method == "GET":
            return render_template("form.html")

        name = (request.form.get("name") or "").strip()
        if not name:
            flash("Name is required.", "error")
            return render_template("form.html", form=request.form), 400

        speaker_id = f"{_slugify(name)}_{uuid.uuid4().hex[:6]}"
        try:
            image_path = _save_image(request.files.get("image"), speaker_id)
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template("form.html", form=request.form), 400

        db.create_speaker({
            "speaker_id": speaker_id,
            "name": name,
            "wikidata_id": (request.form.get("wikidata_id") or "").strip() or None,
            "region": (request.form.get("region") or "").strip() or None,
            "profession": (request.form.get("profession") or "").strip() or None,
            "gender": (request.form.get("gender") or "").strip() or None,
            "notes": (request.form.get("notes") or "").strip() or None,
            "seed_dir": os.path.join("uploads", speaker_id) if image_path else None,
            "image_path": image_path,
            "status": "pending",
        })
        for tag in _split_tags(request.form.get("tags", "")):
            db.add_tag(speaker_id, tag)
        flash(f"Added POI “{name}”.", "success")
        return redirect(url_for("detail", speaker_id=speaker_id))

    @app.route("/poi/<speaker_id>")
    def detail(speaker_id: str):
        """Show one POI with controls to label and tag it."""
        speaker = db.get_speaker(speaker_id)
        if speaker is None:
            abort(404)
        return render_template("detail.html", sp=speaker, statuses=db.STATUSES)

    @app.route("/poi/<speaker_id>/status", methods=["POST"])
    def set_status(speaker_id: str):
        """Update the POI's pipeline status (label)."""
        if db.get_speaker(speaker_id) is None:
            abort(404)
        try:
            db.set_status(speaker_id, request.form.get("status", ""))
            flash("Status updated.", "success")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("detail", speaker_id=speaker_id))

    @app.route("/poi/<speaker_id>/tags/add", methods=["POST"])
    def add_tags(speaker_id: str):
        """Attach one or more comma-separated tags to the POI."""
        if db.get_speaker(speaker_id) is None:
            abort(404)
        for tag in _split_tags(request.form.get("tags", "")):
            db.add_tag(speaker_id, tag)
        return redirect(url_for("detail", speaker_id=speaker_id))

    @app.route("/poi/<speaker_id>/tags/remove", methods=["POST"])
    def remove_tag(speaker_id: str):
        """Detach a single tag from the POI."""
        if db.get_speaker(speaker_id) is None:
            abort(404)
        db.remove_tag(speaker_id, request.form.get("tag", ""))
        return redirect(url_for("detail", speaker_id=speaker_id))

    @app.route("/uploads/<path:relpath>")
    def uploaded_file(relpath: str):
        """Serve a stored reference image."""
        return send_from_directory(app.config["UPLOAD_DIR"], relpath)


def _split_tags(raw: str) -> list[str]:
    """Split a comma-separated tag string into clean, non-empty tags."""
    return [t.strip() for t in raw.split(",") if t.strip()]


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
