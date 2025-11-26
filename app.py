


from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
import os
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from ingestion.utils import read_csv_stream
from ingestion.service import process_records
from ingestion.pandas_cleaner import clean_with_pandas, FORBIDDEN_CHAR_PATTERN
import re
import io
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
from bson.binary import Binary
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
import json
import hashlib
import smtplib
from email.mime.text import MIMEText
import secrets
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = "secret123"
app.permanent_session_lifetime = timedelta(minutes=30)

# Load environment variables from a local .env file if present (development convenience)
load_dotenv()

app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", "587"))
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "")
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "true").lower() != "false"

# MongoDB Config
app.config["MONGO_URI"] = "mongodb://localhost:27017/education_app"
mongo = PyMongo(app)
users = mongo.db.users
courses = mongo.db.courses
enrollments = mongo.db.enrollments
assignments = mongo.db.assignments
submissions = mongo.db.submissions
results = mongo.db.results
academic_records = mongo.db.academic_records
demographics = mongo.db.demographics
lms_events = mongo.db.lms_events
attendance = mongo.db.attendance
feedbacks = mongo.db.feedbacks
announcements = mongo.db.announcements
predictions = mongo.db.predictions
models = mongo.db.models
ml_datasets = mongo.db.ml_datasets
ml_dataset_rows = mongo.db.ml_dataset_rows
# New: separate collections for predictions and notifications
manual_predictions = mongo.db.manual_predictions
ml_predictions = mongo.db.ml_predictions
admin_notifs_col = mongo.db.admin_notifications

# Create helpful indexes (idempotent)
users.create_index("email", unique=True)
users.create_index([("created_at", -1)])
courses.create_index("code", unique=True)
courses.create_index([("instructor_id", 1)])
enrollments.create_index([("user_id", 1), ("course_id", 1)], unique=True)
enrollments.create_index([("course_id", 1), ("status", 1)])
assignments.create_index([("course_id", 1)])
submissions.create_index([("assignment_id", 1), ("student_id", 1)])
results.create_index([("student_id", 1), ("course_id", 1)])
announcements.create_index([("course_id", 1), ("created_at", -1)])
academic_records.create_index([("student_id", 1), ("course_code", 1), ("term", 1)], unique=True)
demographics.create_index("student_id", unique=True)
lms_events.create_index("event_id", unique=True)
attendance.create_index([("student_id", 1), ("course_code", 1), ("date", 1)], unique=True)
feedbacks.create_index([("email", 1), ("created_at", -1)])
# Helpful indexes for predictions listing
try:
    predictions.create_index([("created_at", -1)])
    predictions.create_index([("analyst_email", 1)])
except Exception:
    pass
try:
    models.create_index([("analyst_email", 1), ("created_at", -1)])
    ml_datasets.create_index([("analyst_email", 1), ("created_at", -1)])
    ml_dataset_rows.create_index([("dataset_id", 1)])
    manual_predictions.create_index([("created_at", -1)])
    manual_predictions.create_index([("analyst_email", 1), ("created_at", -1)])
    ml_predictions.create_index([("created_at", -1)])
    ml_predictions.create_index([("analyst_email", 1), ("created_at", -1)])
    admin_notifs_col.create_index([("created_at", -1)])
except Exception:
    pass

# Supported datasets for CSV ingestion/cleaning
# Keep in sync with options in `templates/admin/ingestion.html`
SUPPORTED_DATASETS = {
    "academic_records",
    "demographics",
    "lms",
    "attendance",
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/our-team")
def our_team():
    # Build 6 team members; use current user's name/email if available for the first card
    me_name = session.get("user") or "Muhammad Shariq Baig"
    me_email = session.get("email") or "shariqbaigcena@gmail.com"
    members = [
        {"name": me_name, "email": me_email, "designation": "Project Lead", "image": url_for('static', filename='images/people/Shariq.jpeg')},
        {"name": "Shahzaib", "email": "shehzibaba0031@gmail.com", "designation": "Project Assistant", "image": url_for('static', filename='images/people/shehzi.jpg')},
        {"name": "Saad Siddiqui", "email": "saadshahidsiddiqui@gmail.com", "designation": "Documentation Assistant", "image": url_for('static', filename='images/people/Saad.jpeg')},
        {"name": "Majeed Ahmed", "email": "majeedahmedbrohii@gmail.com", "designation": "Research Assistant", "image": url_for('static', filename='images/people/majeed.jpg')},
        {"name": "Sana Ullah", "email": "anasumrani543@gmail.com", "designation": "UI Reviewer", "image": url_for('static', filename='images/people/sana.jpg')},
        {"name": "Hamza", "email": "minhash299@gmail.com", "designation": "Support Member", "image": url_for('static', filename='images/people/Hamza.jpg')},
    ]
    return render_template("our_team.html", members=members)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    message = None
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip()
        content = (request.form.get("message") or "").strip()
        if content:
            doc = {
                "name": name,
                "email": email,
                "message": content,
                "created_at": datetime.utcnow(),
                "status": "new"
            }
            feedbacks.insert_one(doc)
            message = "Thanks for your feedback! We'll get back to you if needed."
        else:
            message = "Please enter your feedback message."
    return render_template("feedback.html", message=message)

@app.route("/admin/feedback")
def admin_feedback():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    # Mark all 'new' feedback as 'read' now that the admin views the page
    try:
        def resolve_course_display(raw):
            try:
                if not raw:
                    return "UNKNOWN"
                s = str(raw).strip()
                # Try courses collection by _id (ObjectId)
                disp = None
                try:
                    if len(s) == 24:
                        obj = ObjectId(s)
                        c = courses.find_one({"_id": obj})
                        if c:
                            disp = c.get("code") or c.get("title") or c.get("name") or s
                except Exception:
                    pass
                if not disp:
                    # Try raw _id string match or code/title/name
                    c = (courses.find_one({"_id": s}) or
                         courses.find_one({"code": s}) or
                         courses.find_one({"title": s}) or
                         courses.find_one({"name": s}))
                    if c:
                        disp = c.get("code") or c.get("title") or c.get("name") or s
                return (disp or s).strip()
            except Exception:
                return str(raw)

        def resolve_student_display(sid):
            try:
                if not sid:
                    return "Unknown"
                s = str(sid)
                # demographics by id or email
                d = demographics.find_one({"$or": [{"student_id": s}, {"studentId": s}, {"email": s}]}) or {}
                name = d.get("name") or d.get("full_name") or d.get("email")
                if name:
                    return name
                # users by _id
                try:
                    if len(s) == 24:
                        u = users.find_one({"_id": ObjectId(s)})
                        if u:
                            return u.get("name") or u.get("email") or s
                except Exception:
                    pass
                # users by email
                u = users.find_one({"email": s})
                if u:
                    return u.get("name") or u.get("email") or s
                return s
            except Exception:
                return str(sid)
        feedbacks.update_many({"status": "new"}, {"$set": {"status": "read", "read_at": datetime.utcnow()}})
    except Exception:
        pass
    items = list(feedbacks.find().sort("created_at", -1))
    return render_template("admin/feedback.html", user=session.get("user"), role="Admin", items=items)


@app.route("/admin/feedback/<fid>/delete", methods=["POST"])
def admin_feedback_delete(fid):
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    deleted = 0
    try:
        oid = ObjectId(fid)
        res = feedbacks.delete_one({"_id": oid})
        deleted = res.deleted_count
    except Exception:
        deleted = 0
    if deleted:
        flash("Feedback deleted.", "success")
    else:
        flash("Could not delete feedback.", "warning")
    next_url = request.form.get("next") or url_for("admin_feedback")
    return redirect(next_url)

# Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        existing_user = users.find_one({"email": email})
        if existing_user:
            flash("Email already registered! Please login instead.", "warning")
            return redirect(url_for("login"))

        new_user = {"name": name, "email": email, "password": password, "role": role, "created_at": datetime.utcnow()}
        users.insert_one(new_user)
        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users.find_one({"email": email, "password": password})
        if user:
            session.permanent = True
            session["user"] = user["name"]
            session["role"] = user["role"]
            session["email"] = user["email"]
            # Store user_id as string for convenience
            session["user_id"] = str(user.get("_id"))

            flash(f"Welcome back, {user['name']}!", "success")

            if user["role"] == "Admin":
                return redirect(url_for("admin_dashboard"))
            elif user["role"] == "Teacher":
                return redirect(url_for("teacher_dashboard"))
            elif user["role"] == "Student":
                return redirect(url_for("student_dashboard"))
            elif user["role"] == "Analyst":
                return redirect(url_for("analyst_dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            return render_template("login.html")
    return render_template("login.html")

# Forgot Password - Request Reset (send 6-digit code)
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        if not email:
            flash("Please enter your email.", "warning")
            return render_template("forgot_password.html")
        user = users.find_one({"email": email})
        # Always respond the same to avoid email enumeration
        code = f"{secrets.randbelow(1000000):06d}"
        expires = datetime.utcnow() + timedelta(minutes=15)
        if user:
            users.update_one({"_id": user["_id"]}, {"$set": {"reset_code": code, "reset_expires": expires}, "$unset": {"reset_token": ""}})

        sent = False
        try:
            smtp_host = app.config.get("MAIL_SERVER")
            smtp_port = int(app.config.get("MAIL_PORT", 587))
            smtp_user = app.config.get("MAIL_USERNAME")
            smtp_pass = app.config.get("MAIL_PASSWORD")
            use_tls = app.config.get("MAIL_USE_TLS", True)
            if smtp_host and smtp_user and smtp_pass and user:
                msg = MIMEText(f"Your password reset code is: {code}\nThis code expires in 15 minutes.")
                msg["Subject"] = "Your Password Reset Code"
                msg["From"] = smtp_user
                msg["To"] = email
                server = smtplib.SMTP(smtp_host, smtp_port)
                if use_tls:
                    server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, [email], msg.as_string())
                server.quit()
                sent = True
        except Exception:
            sent = False

        if sent:
            flash("A verification code has been sent to your email.", "success")
        else:
            # Do not leak codes in UI; keep response generic to avoid enumeration
            flash("If an account exists for that email, a verification code has been sent.", "info")
        return redirect(url_for("reset_password_code"))

    return render_template("forgot_password.html")

# Forgot Password - Reset with token (kept for compatibility)
@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    now = datetime.utcnow()
    user = users.find_one({"reset_token": token, "reset_expires": {"$gt": now}})
    if not user:
        flash("Invalid or expired reset link.", "danger")
        return redirect(url_for("forgot_password"))
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        confirm = (request.form.get("confirm") or "").strip()
        if not password:
            flash("Please enter a new password.", "warning")
            return render_template("reset_password.html")
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("reset_password.html")
        users.update_one({"_id": user["_id"]}, {"$set": {"password": password}, "$unset": {"reset_token": "", "reset_expires": ""}})
        flash("Your password has been reset. Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("reset_password.html")

# Forgot Password - Reset using email + code
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password_code():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        code = (request.form.get("code") or "").strip()
        new_password = (request.form.get("new_password") or "").strip()
        confirm_password = (request.form.get("confirm_password") or "").strip()
        if not email or not code or not new_password or not confirm_password:
            flash("Please fill in all fields.", "warning")
            return render_template("reset_password.html", email=email)
        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("reset_password.html", email=email)
        now = datetime.utcnow()
        user = users.find_one({"email": email, "reset_code": code, "reset_expires": {"$gt": now}})
        if not user:
            flash("Invalid or expired code.", "danger")
            return render_template("reset_password.html", email=email)
        users.update_one({"_id": user["_id"]}, {"$set": {"password": new_password}, "$unset": {"reset_code": "", "reset_expires": "", "reset_token": ""}})
        flash("Your password has been reset. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("reset_password.html")

# Dashboards
@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "Admin":
        return redirect(url_for("home"))
    # Show recent feedback and new feedback count on dashboard
    recent_feedback = list(feedbacks.find().sort("created_at", -1).limit(5))
    new_count = feedbacks.count_documents({"status": "new"})
    total_users = users.count_documents({})

    # Manual predictions (latest few)
    try:
        recent_manual_preds = list(manual_predictions.find().sort("created_at", -1).limit(5))
        manual_pred_count = manual_predictions.count_documents({})
    except Exception:
        recent_manual_preds = []
        manual_pred_count = 0
    # Notifications for model training
    try:
        recent_notifications = list(admin_notifs_col.find({"type": "model_trained"}).sort("created_at", -1).limit(5))
        notifications_count = admin_notifs_col.count_documents({"type": "model_trained"})
    except Exception:
        recent_notifications = []
        notifications_count = 0

    # Build dashboard lists
    # Students with enrolled (active) course codes
    student_rows = []
    debug_mode = (request.args.get('debug') == '1')
    debug_students = []
    try:
        # Show the most recently created students (fallback to _id) so new signups appear on the dashboard
        for stu in users.find({"role": "Student"}).sort([("created_at", -1), ("_id", -1)]).limit(20):
            sid = str(stu.get("_id"))
            sid_obj = stu.get("_id")
            codes = []
            # Match enrollments by student id (several possible field names). Do not filter by status to be more inclusive.
            stu_email = (stu.get("email") or None)
            enr_q = {
                "$or": [
                    {"student_id": sid}, {"student_id": sid_obj},
                    {"studentId": sid}, {"studentId": sid_obj},
                    {"user_id": sid}, {"userId": sid},
                    # Also match by email if enrollments store it instead of ids
                    ({"student_email": stu_email} if stu_email else {}),
                    ({"email": stu_email} if stu_email else {}),
                ]
            }
            found_enrs = list(enrollments.find(enr_q).limit(10))
            for e in found_enrs:
                cid = e.get("course_id") or e.get("courseId") or e.get("course")
                c = None
                if cid:
                    try:
                        if isinstance(cid, str) and len(cid) == 24:
                            # Try as ObjectId first
                            try:
                                c = courses.find_one({"_id": ObjectId(cid)})
                            except Exception:
                                c = None
                            # If not found, try as raw string _id
                            if c is None:
                                c = courses.find_one({"_id": cid})
                        elif isinstance(cid, str):
                            # Non-ObjectId string, try as raw _id
                            c = courses.find_one({"_id": cid})
                        else:
                            # Already an ObjectId or other type
                            c = courses.find_one({"_id": cid})
                    except Exception:
                        c = None
                    if c is None and isinstance(cid, str):
                        # Fallbacks by known course fields
                        c = (courses.find_one({"code": cid}) or
                             courses.find_one({"title": cid}) or
                             courses.find_one({"name": cid}))
                
                # Only add course code if the course still exists
                if c is not None:
                    code = (c.get("code") or c.get("title") or c.get("name"))
                    if code and code not in codes:  # Avoid duplicates
                        codes.append(code)
                # If course doesn't exist, we won't add it to the codes list
                # This ensures deleted courses don't show up in the admin dashboard
            # If still empty, try alternate course schemas that embed student references
            if not codes:
                alt_or = []
                try:
                    stu_email = stu.get("email")
                except Exception:
                    stu_email = None
                # Arrays of ids or emails
                if sid:
                    alt_or.append({"enrolled_student_ids": {"$in": [sid]}})
                    alt_or.append({"students": {"$in": [sid]}})
                    alt_or.append({"student_ids": {"$in": [sid]}})
                if sid_obj:
                    alt_or.append({"enrolled_student_ids": {"$in": [sid_obj]}})
                    alt_or.append({"students": {"$in": [sid_obj]}})
                    alt_or.append({"student_ids": {"$in": [sid_obj]}})
                if stu_email:
                    alt_or.append({"enrolled_emails": {"$in": [stu_email]}})
                    alt_or.append({"students": {"$in": [stu_email]}})
                if alt_or:
                    for cdoc in courses.find({"$or": alt_or}).limit(10):
                        disp = cdoc.get("code") or cdoc.get("title") or cdoc.get("name")
                        if disp:
                            codes.append(disp)
            # Final fallback: infer from results documents for this student
            if not codes:
                try:
                    r_q = {"$or": []}
                    r_q["$or"].append({"student_id": sid})
                    r_q["$or"].append({"user_id": sid})
                    if stu.get("email"):
                        r_q["$or"].append({"student_email": stu.get("email")})
                        r_q["$or"].append({"email": stu.get("email")})
                    for rdoc in results.find(r_q).limit(20):
                        disp = rdoc.get("course_id")
                        if disp:
                            codes.append(disp)
                except Exception:
                    pass
            # Deduplicate course labels for the student
            try:
                codes = sorted(set([c for c in codes if c]))
            except Exception:
                pass
            sr = {
                "name": stu.get("name"),
                "email": stu.get("email"),
                "courses": codes,
            }
            student_rows.append(sr)
            if debug_mode:
                debug_students.append({
                    "student": sr,
                    "enrollments": [{k: v for k, v in doc.items() if k in ("_id","student_id","studentId","status","course_id","courseId","course","course_code","course_title")} for doc in found_enrs]
                })
    except Exception:
        student_rows = []

    # Teachers with offered courses (by course code)
    teacher_rows = []
    debug_teachers = []
    try:
        for t in users.find({"role": "Teacher"}).sort("name", 1).limit(8):
            tid = t.get("_id")
            codes = []
            tids = [tid, str(tid) if tid else None]
            or_parts = []
            for v in tids:
                if v is not None:
                    or_parts.extend([
                        {"teacher_id": v}, {"teacherId": v}, {"instructorId": v}, {"instructor_id": v}
                    ])
            t_email = t.get("email")
            t_name = t.get("name")
            if t_email:
                or_parts.append({"teacher_email": t_email})
                or_parts.append({"instructor_email": t_email})
            if t_name:
                or_parts.append({"teacher_name": t_name})
                or_parts.append({"instructor_name": t_name})
            qry = {"$or": or_parts} if or_parts else {}
            primary_courses = list(courses.find(qry).limit(10))
            for c in primary_courses:
                codes.append(c.get("code") or c.get("title") or c.get("name"))
            if not codes:
                extra_or = []
                if t_email:
                    extra_or.extend([{"owner_email": t_email}, {"creator_email": t_email}])
                if t_name:
                    extra_or.extend([{"owner_name": t_name}, {"instructor": t_name}, {"teacher": t_name}])
                if extra_or:
                    for c in courses.find({"$or": extra_or}):
                        disp = c.get("code") or c.get("title") or c.get("name")
                        if disp:
                            codes.append(disp)
            if not codes:
                try:
                    a_or = [{"teacher_id": v} for v in tids if v is not None]
                    if t_email:
                        a_or.append({"teacher_email": t_email})
                    if t_name:
                        a_or.append({"teacher_name": t_name})
                    for a in assignments.find({"$or": a_or}):
                        ac = None
                        acid = a.get("course_id")
                        if acid:
                            try:
                                if isinstance(acid, str) and len(acid) == 24:
                                    ac = courses.find_one({"_id": ObjectId(acid)})
                                else:
                                    ac = courses.find_one({"_id": acid})
                            except Exception:
                                ac = None
                        if ac is None and isinstance(acid, str):
                            ac = courses.find_one({"code": acid}) or courses.find_one({"title": acid})
                        disp = (a.get("course_code") or (ac or {}).get("code") or (ac or {}).get("title") or (ac or {}).get("name"))
                        if disp:
                            codes.append(disp)
                except Exception:
                    pass
            try:
                codes = sorted(set([c for c in codes if c]))
            except Exception:
                pass
            tr = {
                "name": t.get("name"),
                "email": t.get("email"),
                "courses": [x for x in codes if x],
            }
            teacher_rows.append(tr)
            # Collect debug info
            if debug_mode:
                debug_teachers.append({
                    "teacher": tr,
                    "primary_query": qry,
                    "primary_courses_len": len(primary_courses) if 'primary_courses' in locals() else 0,
                })
    except Exception:
        teacher_rows = []

    # Analysts list
    analyst_rows = []
    try:
        for a in users.find({"role": "Analyst"}).sort("name", 1).limit(8):
            analyst_rows.append({
                "name": a.get("name"),
                "email": a.get("email"),
            })
    except Exception:
        analyst_rows = []

    # Recent analyst predictions
    recent_predictions = []
    try:
        recent_predictions = list(predictions.find().sort("created_at", -1).limit(10))
    except Exception:
        recent_predictions = []

    return render_template(
        "admin_dashboard.html",
        user=session.get("user"),
        role="Admin",
        recent_feedback=recent_feedback,
        new_feedback_count=new_count,
        total_users=total_users,
        recent_manual_preds=recent_manual_preds,
        manual_pred_count=manual_pred_count,
        recent_notifications=recent_notifications,
        notifications_count=notifications_count,
        student_rows=student_rows,
        teacher_rows=teacher_rows,
        analyst_rows=analyst_rows,
        recent_predictions=recent_predictions,
        debug_students=debug_students if debug_mode else None,
        debug_teachers=debug_teachers if debug_mode else None,
    )


# -------------------------
# Admin: Full Lists
# -------------------------

@app.route("/admin/students")
def admin_students_overview():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    rows = []
    try:
        for stu in users.find({"role": "Student"}).sort("name", 1):
            sid = str(stu.get("_id"))
            sid_obj = stu.get("_id")
            codes = []
            enr_q = {"$or": [
                {"student_id": sid}, {"student_id": sid_obj},
                {"studentId": sid}, {"studentId": sid_obj},
                {"user_id": sid}, {"userId": sid},
            ]}
            for e in enrollments.find(enr_q):
                cid = e.get("course_id") or e.get("courseId") or e.get("course")
                c = None
                if cid:
                    try:
                        if isinstance(cid, str) and len(cid) == 24:
                            # Try as ObjectId first
                            try:
                                c = courses.find_one({"_id": ObjectId(cid)})
                            except Exception:
                                c = None
                            # If not found, try as raw string _id
                            if c is None:
                                c = courses.find_one({"_id": cid})
                        elif isinstance(cid, str):
                            # Non-ObjectId string, try as raw _id
                            c = courses.find_one({"_id": cid})
                        else:
                            # Already an ObjectId or other type
                            c = courses.find_one({"_id": cid})
                    except Exception:
                        c = None
                    if c is None and isinstance(cid, str):
                        # Fallbacks by known course fields
                        c = (courses.find_one({"code": cid}) or
                             courses.find_one({"title": cid}) or
                             courses.find_one({"name": cid}))
                label = ((c or {}).get("code") or (c or {}).get("title") or (c or {}).get("name") or
                         e.get("course_code") or e.get("course_title") or e.get("course") or e.get("code") or e.get("name"))
                if label:
                    codes.append(label)
            # results fallback
            if not codes:
                try:
                    r_q = {"$or": [{"student_id": sid}, {"user_id": sid}]}
                    if stu.get("email"):
                        r_q["$or"].extend([{"student_email": stu.get("email")}, {"email": stu.get("email")}])
                    for rdoc in results.find(r_q):
                        disp = rdoc.get("course_id")
                        if disp:
                            codes.append(disp)
                except Exception:
                    pass
            try:
                codes = sorted(set([c for c in codes if c]))
            except Exception:
                pass
            rows.append({"name": stu.get("name"), "email": stu.get("email"), "courses": codes})
    except Exception:
        rows = []
    return render_template("admin/students_overview.html", user=session.get("user"), role="Admin", rows=rows)


@app.route("/admin/teachers")
def admin_teachers_overview():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    rows = []
    try:
        for t in users.find({"role": "Teacher"}).sort("name", 1):
            tid = t.get("_id")
            codes = []
            tids = [tid, str(tid) if tid else None]
            or_parts = []
            for v in tids:
                if v is not None:
                    or_parts.extend([
                        {"teacher_id": v}, {"teacherId": v}, {"instructorId": v}, {"instructor_id": v}
                    ])
            t_email = t.get("email")
            t_name = t.get("name")
            if t_email:
                or_parts.append({"teacher_email": t_email})
                or_parts.append({"instructor_email": t_email})
            if t_name:
                or_parts.append({"teacher_name": t_name})
                or_parts.append({"instructor_name": t_name})
            qry = {"$or": or_parts} if or_parts else {}
            for c in courses.find(qry):
                disp = c.get("code") or c.get("title") or c.get("name")
                if disp:
                    codes.append(disp)
            if not codes:
                extra_or = []
                if t_email:
                    extra_or.extend([{"owner_email": t_email}, {"creator_email": t_email}])
                if t_name:
                    extra_or.extend([{"owner_name": t_name}, {"instructor": t_name}, {"teacher": t_name}])
                if extra_or:
                    for c in courses.find({"$or": extra_or}):
                        disp = c.get("code") or c.get("title") or c.get("name")
                        if disp:
                            codes.append(disp)
            if not codes:
                try:
                    a_or = [{"teacher_id": v} for v in tids if v is not None]
                    if t_email:
                        a_or.append({"teacher_email": t_email})
                    if t_name:
                        a_or.append({"teacher_name": t_name})
                    for a in assignments.find({"$or": a_or}):
                        ac = None
                        acid = a.get("course_id")
                        if acid:
                            try:
                                if isinstance(acid, str) and len(acid) == 24:
                                    ac = courses.find_one({"_id": ObjectId(acid)})
                                else:
                                    ac = courses.find_one({"_id": acid})
                            except Exception:
                                ac = None
                        if ac is None and isinstance(acid, str):
                            ac = courses.find_one({"code": acid}) or courses.find_one({"title": acid})
                        disp = (a.get("course_code") or (ac or {}).get("code") or (ac or {}).get("title") or (ac or {}).get("name"))
                        if disp:
                            codes.append(disp)
                except Exception:
                    pass
            try:
                codes = sorted(set([c for c in codes if c]))
            except Exception:
                pass
            rows.append({"name": t.get("name"), "email": t.get("email"), "courses": codes})
    except Exception:
        rows = []
    return render_template("admin/teachers_overview.html", user=session.get("user"), role="Admin", rows=rows)


@app.route("/admin/analysts")
def admin_analysts_overview():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    rows = []
    try:
        for a in users.find({"role": "Analyst"}).sort("name", 1):
            rows.append({"name": a.get("name"), "email": a.get("email")})
    except Exception:
        rows = []
    return render_template("admin/analysts_overview.html", user=session.get("user"), role="Admin", rows=rows)


@app.route("/admin/predictions")
def admin_predictions():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    analyst = (request.args.get("analyst") or "").strip()
    email = (request.args.get("email") or "").strip()
    start = (request.args.get("start") or "").strip()
    end = (request.args.get("end") or "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except Exception:
        page = 1
    try:
        per_page = min(50, max(5, int(request.args.get("per_page", 10))))
    except Exception:
        per_page = 10
    q = {}
    if analyst:
        q["analyst"] = {"$regex": analyst, "$options": "i"}
    if email:
        q["analyst_email"] = {"$regex": email, "$options": "i"}
    date_filter = {}
    try:
        if start:
            sd = datetime.fromisoformat(start + "T00:00:00")
            date_filter["$gte"] = sd
        if end:
            ed = datetime.fromisoformat(end + "T23:59:59")
            date_filter["$lte"] = ed
    except Exception:
        date_filter = {}
    if date_filter:
        q["created_at"] = date_filter
    try:
        total = manual_predictions.count_documents(q)
        items = list(manual_predictions.find(q).sort("created_at", -1).skip((page-1)*per_page).limit(per_page))
    except Exception:
        total = 0
        items = []
    pages = (total + per_page - 1) // per_page if per_page else 1
    return render_template(
        "admin/predictions.html",
        user=session.get("user"), role="Admin",
        items=items, total=total, page=page, per_page=per_page, pages=pages,
        analyst=analyst, email=email, start=start, end=end,
    )


@app.route("/admin/feedback_count")
def admin_feedback_count():
    if session.get("role") != "Admin":
        return jsonify({"error": "unauthorized"}), 403
    new_count = feedbacks.count_documents({"status": "new"})
    latest = feedbacks.find().sort("created_at", -1).limit(1)
    latest_doc = next(latest, None)
    latest_ts = latest_doc.get("created_at").isoformat() if latest_doc and latest_doc.get("created_at") else None
    return jsonify({"new_count": new_count, "latest_created_at": latest_ts})

@app.route("/admin/user_count")
def admin_user_count():
    if session.get("role") != "Admin":
        return jsonify({"error": "unauthorized"}), 403
    total = users.count_documents({})
    latest = users.find({"created_at": {"$exists": True}}).sort("created_at", -1).limit(1)
    latest_doc = next(latest, None)
    latest_ts = latest_doc.get("created_at").isoformat() if latest_doc and latest_doc.get("created_at") else None
    return jsonify({"total": total, "latest_created_at": latest_ts})

@app.route("/teacher")
def teacher_dashboard():
    if session.get("role") != "Teacher":
        return redirect(url_for("home"))
    return render_template("teacher_dashboard.html", user=session.get("user"), role="Teacher")


# -------------------------
# Student model metadata and predict (read-only)
# -------------------------

@app.route("/api/student/model/meta")
def api_student_model_meta():
    # Anyone logged-in as Student can view meta; avoid exposing analyst info
    if session.get("role") != "Student":
        return jsonify({"error": "unauthorized"}), 403
    try:
        mdl = models.find_one({}, sort=[("created_at", -1)])
        if not mdl:
            return jsonify({"error": "No trained model available"}), 404
        cols = mdl.get("feature_columns", [])
        fields = []
        encoders = {}
        try:
            blob = mdl.get("encoders_blob")
            if blob:
                encoders = pickle.loads(blob) or {}
        except Exception:
            encoders = {}
        for c in cols:
            f = {"name": c}
            if c in encoders:
                try:
                    classes = list(getattr(encoders[c], "classes_", []))
                except Exception:
                    classes = []
                f.update({"type": "categorical", "options": classes})
            else:
                f.update({"type": "number"})
            fields.append(f)
        return jsonify({
            "feature_columns": cols,
            "fields": fields,
            "target": mdl.get("target"),
            "type": mdl.get("type")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Student: multiple model metas (unique per target) and predict by model id
# -------------------------

@app.route("/api/student/models/meta")
def api_student_models_meta():
    if session.get("role") != "Student":
        return jsonify({"error": "unauthorized"}), 403
    try:
        seen = set()
        out = []
        cursor = models.find({}).sort("created_at", -1).limit(200)
        for mdl in cursor:
            tgt = (mdl.get("target") or "").strip()
            if not tgt:
                continue
            key = tgt.lower()
            # Exclude unwanted targets from student UI
            if key in ("predicted_category",):
                continue
            if key in seen:
                continue
            seen.add(key)
            cols = mdl.get("feature_columns", []) or []
            fields = []
            encoders = {}
            try:
                blob = mdl.get("encoders_blob")
                if blob:
                    encoders = pickle.loads(blob) or {}
            except Exception:
                encoders = {}
            for c in cols:
                cname = str(c).strip()
                if cname.lower() in ("student_id", "studentid", "id"):
                    continue
                f = {"name": c}
                if c in encoders:
                    try:
                        classes = list(getattr(encoders[c], "classes_", []))
                    except Exception:
                        classes = []
                    f.update({"type": "categorical", "options": classes})
                else:
                    f.update({"type": "number"})
                fields.append(f)
            out.append({
                "model_id": str(mdl.get("_id")),
                "target": mdl.get("target"),
                "type": mdl.get("type"),
                "feature_columns": cols,
                "fields": fields,
            })
        if not out:
            return jsonify({"error": "No trained models available"}), 404
        return jsonify({"models": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/student/model/<mid>/predict", methods=["POST"])
def api_student_model_predict_by_id(mid):
    if session.get("role") != "Student":
        return jsonify({"error": "unauthorized"}), 403
    try:
        payload = request.json or {}
        try:
            oid = ObjectId(mid)
        except Exception:
            return jsonify({"error": "invalid model id"}), 400
        mdl = models.find_one({"_id": oid})
        if not mdl:
            return jsonify({"error": "Model not found"}), 404
        model_cols = mdl.get("feature_columns", []) or []
        row = {col: payload.get(col) for col in model_cols}
        df_in = pd.DataFrame([row])
        encoders = {}
        try:
            blob = mdl.get("encoders_blob")
            if blob:
                encoders = pickle.loads(blob) or {}
        except Exception:
            encoders = {}
        for col in model_cols:
            if col in encoders:
                val = str(df_in.at[0, col]) if col in df_in.columns else ""
                classes = list(getattr(encoders[col], "classes_", []))
                try:
                    idx = classes.index(val)
                except ValueError:
                    idx = 0
                df_in[col] = idx
            else:
                df_in[col] = pd.to_numeric(df_in.get(col, 0), errors="coerce").fillna(0)
        mtype = mdl.get("type")
        if mtype == "logistic_regression":
            coef = np.array(mdl.get("coef", []), dtype=float).ravel()
            intercept = float(mdl.get("intercept", 0.0))
            linear = float(np.dot(df_in[model_cols].values[0], coef) + intercept)
            prob = 1.0 / (1.0 + np.exp(-linear))
            label = 1 if prob >= 0.5 else 0
            return jsonify({
                "prediction": label,
                "probability": round(float(prob), 3),
                "target": mdl.get("target"),
                "is_binary": True,
            })
        if mtype == "random_forest" and mdl.get("model_blob") is not None:
            rf = pickle.loads(mdl["model_blob"]) or None
            y_pred = rf.predict(df_in[model_cols].values)[0]
            prob = None
            try:
                proba = rf.predict_proba(df_in[model_cols].values)[0]
                prob = float(np.max(proba))
            except Exception:
                prob = None
            # Map back to original label if y_classes stored
            y_classes = mdl.get("y_classes") or []
            try:
                if isinstance(y_pred, (int, np.integer)) and y_classes and int(y_pred) < len(y_classes):
                    pred_out = str(y_classes[int(y_pred)])
                else:
                    pred_out = (int(y_pred) if isinstance(y_pred, (int, np.integer)) else y_pred)
            except Exception:
                pred_out = (int(y_pred) if isinstance(y_pred, (int, np.integer)) else y_pred)
            return jsonify({
                "prediction": pred_out,
                "probability": (round(float(prob), 3) if isinstance(prob, (int, float)) else None),
                "target": mdl.get("target"),
                "is_binary": bool(mdl.get("is_binary")),
            })
        # default: linear regression numeric
        coef = np.array(mdl.get("coef", []), dtype=float).ravel()
        intercept = float(mdl.get("intercept", 0.0))
        y = float(np.dot(df_in[model_cols].values[0], coef) + intercept)
        is_bin = bool(mdl.get("is_binary"))
        pred_val = 1 if (1.0/(1.0+np.exp(-y)) >= 0.5) else 0 if is_bin else round(y, 4)
        if is_bin:
            return jsonify({
                "prediction": int(pred_val),
                "probability": round(float(1.0/(1.0+np.exp(-y))), 3),
                "target": mdl.get("target"),
                "is_binary": True,
            })
        else:
            return jsonify({
                "prediction": round(y, 4),
                "probability": None,
                "target": mdl.get("target"),
                "is_binary": False,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/student/model/predict", methods=["POST"])
def api_student_model_predict():
    if session.get("role") != "Student":
        return jsonify({"error": "unauthorized"}), 403
    try:
        payload = request.json or {}
        mdl = models.find_one({}, sort=[("created_at", -1)])
        if not mdl:
            return jsonify({"error": "No trained model available"}), 404
        model_cols = mdl.get("feature_columns", []) or []
        row = {col: payload.get(col) for col in model_cols}
        df_in = pd.DataFrame([row])
        encoders = {}
        try:
            blob = mdl.get("encoders_blob")
            if blob:
                encoders = pickle.loads(blob) or {}
        except Exception:
            encoders = {}
        for col in model_cols:
            if col in encoders:
                val = str(df_in.at[0, col]) if col in df_in.columns else ""
                classes = list(getattr(encoders[col], "classes_", []))
                try:
                    idx = classes.index(val)
                except ValueError:
                    idx = 0
                df_in[col] = idx
            else:
                df_in[col] = pd.to_numeric(df_in.get(col, 0), errors="coerce").fillna(0)

        mtype = mdl.get("type")
        if mtype == "logistic_regression":
            coef = np.array(mdl.get("coef", []), dtype=float).ravel()
            intercept = float(mdl.get("intercept", 0.0))
            linear = float(np.dot(df_in[model_cols].values[0], coef) + intercept)
            prob = 1.0 / (1.0 + np.exp(-linear))
            label = 1 if prob >= 0.5 else 0
            return jsonify({
                "prediction": label,
                "probability": round(float(prob), 3),
                "target": mdl.get("target"),
                "used_features": model_cols,
            })
        if mtype == "random_forest" and mdl.get("model_blob") is not None:
            rf = pickle.loads(mdl["model_blob"]) or None
            y_pred = rf.predict(df_in[model_cols].values)[0]
            prob = None
            try:
                proba = rf.predict_proba(df_in[model_cols].values)[0]
                prob = float(max(proba))
            except Exception:
                prob = None
            return jsonify({
                "prediction": int(y_pred) if isinstance(y_pred, (int, np.integer)) else y_pred,
                "probability": (round(float(prob), 3) if isinstance(prob, (int, float)) else None),
                "target": mdl.get("target"),
                "used_features": model_cols,
                "is_binary": bool(mdl.get("is_binary")),
            })
        # default: linear regression style
        coef = np.array(mdl.get("coef", []), dtype=float).ravel()
        intercept = float(mdl.get("intercept", 0.0))
        y = float(np.dot(df_in[model_cols].values[0], coef) + intercept)
        return jsonify({
            "prediction": round(y, 4),
            "probability": None,
            "target": mdl.get("target"),
            "used_features": model_cols,
            "is_binary": bool(mdl.get("is_binary")),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Teacher: Course Management
# -------------------------

def require_teacher():
    if session.get("role") != "Teacher":
        return False
    return True

@app.route("/teacher/courses", methods=["GET"]) 
def teacher_courses():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    items = list(courses.find({"instructor_id": ObjectId(instructor_id)}).sort("code", 1)) if instructor_id else []
    return render_template("teacher/courses.html", user=session.get("user"), role="Teacher", items=items)


@app.route("/teacher/courses/create", methods=["POST"]) 
def teacher_courses_create():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    code = (request.form.get("code") or "").strip().upper()
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    if not code or not title:
        flash("Code and Title are required.", "warning")
        return redirect(url_for("teacher_courses"))
    # If course with code exists but no instructor, allow teacher to claim by setting instructor_id if not set
    existing = courses.find_one({"code": code})
    if existing and existing.get("instructor_id") and str(existing.get("instructor_id")) != instructor_id:
        flash("Course code already owned by another instructor.", "danger")
        return redirect(url_for("teacher_courses"))
    if existing:
        courses.update_one({"_id": existing["_id"]}, {"$set": {
            "title": title,
            "description": description,
            "active": True,
            "instructor_id": ObjectId(instructor_id),
        }})
        flash("Course updated.", "success")
    else:
        courses.insert_one({
            "code": code,
            "title": title,
            "description": description,
            "active": True,
            "created_at": datetime.utcnow(),
            "instructor_id": ObjectId(instructor_id),
        })
        flash("Course created.", "success")
    return redirect(url_for("teacher_courses"))


@app.route("/teacher/courses/<cid>/update", methods=["POST"]) 
def teacher_courses_update(cid):
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    try:
        oid = ObjectId(cid)
    except Exception:
        flash("Invalid course id.", "danger")
        return redirect(url_for("teacher_courses"))
    doc = courses.find_one({"_id": oid})
    if not doc or str(doc.get("instructor_id")) != instructor_id:
        flash("Not authorized to update this course.", "danger")
        return redirect(url_for("teacher_courses"))
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    update_doc = {}
    # Only update 'active' if explicitly sent by the form
    if "active" in request.form:
        active = (request.form.get("active") or "on") in ("on", "true", "1", "yes")
        update_doc["active"] = active
    if title:
        update_doc["title"] = title
    update_doc["description"] = description
    courses.update_one({"_id": oid}, {"$set": update_doc})
    flash("Course updated.", "success")
    return redirect(url_for("teacher_courses"))


@app.route("/teacher/courses/<cid>/delete", methods=["POST"]) 
def teacher_courses_delete(cid):
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    try:
        oid = ObjectId(cid)
    except Exception:
        flash("Invalid course id.", "danger")
        return redirect(url_for("teacher_courses"))
    doc = courses.find_one({"_id": oid})
    if not doc or str(doc.get("instructor_id")) != instructor_id:
        flash("Not authorized to delete this course.", "danger")
        return redirect(url_for("teacher_courses"))
    
    # Get course code before deletion for cleaning up related data
    course_code = doc.get("code")
    
    # Delete all assignments and their submissions for this course
    assignment_ids = [a["_id"] for a in assignments.find({"course_id": oid}, {"_id": 1})]
    if assignment_ids:
        submissions.delete_many({"assignment_id": {"$in": assignment_ids}})
        assignments.delete_many({"_id": {"$in": assignment_ids}})
    
    # Delete all results for this course
    results.delete_many({"course_id": oid})
    
    # Delete all enrollments for this course
    enrollments.delete_many({"course_id": oid})
    
    # Delete all announcements for this course
    announcements.delete_many({"course_id": oid})
    
    # Delete attendance records for this course
    attendance.delete_many({"course_code": course_code})
    
    # Delete the course itself
    courses.delete_one({"_id": oid})
    
    flash("Course and all related data have been deleted.", "success")
    return redirect(url_for("teacher_courses"))


# -------------------------
# Teacher: Student Management
# -------------------------

@app.route("/teacher/students")
def teacher_students():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    if not instructor_id:
        return redirect(url_for("login"))
    # Find teacher-owned courses
    owned = list(courses.find({"instructor_id": ObjectId(instructor_id)}))
    course_map = {str(c["_id"]): c for c in owned}
    course_ids = [c["_id"] for c in owned]
    # Fetch enrollments for these courses
    enr_list = list(enrollments.find({"course_id": {"$in": course_ids}}).sort("status", 1)) if course_ids else []
    # Build student info map
    student_ids = list({e.get("user_id") for e in enr_list if e.get("user_id")})
    student_oid_map = {}
    for sid in student_ids:
        try:
            student_oid_map[sid] = ObjectId(sid)
        except Exception:
            continue
    students_info = {}
    if student_oid_map:
        docs = users.find({"_id": {"$in": list(student_oid_map.values())}})
        for d in docs:
            students_info[str(d["_id"])] = d
    # Prepare view model
    rows = []
    for e in enr_list:
        c = course_map.get(str(e.get("course_id")))
        s = students_info.get(e.get("user_id"))
        rows.append({
            "enrollment_id": str(e.get("_id")),
            "course_code": (c.get("code") if c else e.get("course_code")),
            "course_title": (c.get("title") if c else ""),
            "student_name": s.get("name") if s else "(Unknown)",
            "student_email": s.get("email") if s else "",
            "status": e.get("status") or "active",
        })
    return render_template("teacher/students.html", user=session.get("user"), role="Teacher", rows=rows)


@app.route("/teacher/enrollments/<eid>/approve", methods=["POST"]) 
def teacher_enrollment_approve(eid):
    if not require_teacher():
        return redirect(url_for("login"))
    # Set status to active
    try:
        oid = ObjectId(eid)
        enrollments.update_one({"_id": oid}, {"$set": {"status": "active"}})
        flash("Enrollment approved.", "success")
    except Exception:
        flash("Could not approve enrollment.", "danger")
    return redirect(url_for("teacher_students"))


# -------------------------
# Teacher: Assignments CRUD
# -------------------------

@app.route("/teacher/assignments")
def teacher_assignments():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    owned = list(courses.find({"instructor_id": ObjectId(instructor_id)})) if instructor_id else []
    course_ids = [c["_id"] for c in owned]
    course_map = {str(c["_id"]): c for c in owned}
    items = []
    if course_ids:
        for a in assignments.find({"course_id": {"$in": course_ids}}).sort("_id", -1):
            c = course_map.get(str(a.get("course_id")))
            items.append({
                "_id": str(a.get("_id")),
                "title": a.get("title"),
                "description": a.get("description"),
                "deadline": a.get("deadline"),
                "attachment_name": a.get("attachment_name"),
                "course_code": c.get("code") if c else "",
                "course_title": c.get("title") if c else "",
            })
    # For create form
    simple_courses = [{"_id": str(c["_id"]), "code": c.get("code"), "title": c.get("title")} for c in owned]
    return render_template("teacher/assignments.html", user=session.get("user"), role="Teacher", items=items, courses=simple_courses)


@app.route("/teacher/assignments/create", methods=["POST"]) 
def teacher_assignments_create():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    course_id = request.form.get("course_id")
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    deadline = (request.form.get("deadline") or "").strip()
    attachment = request.files.get("attachment")
    attachment_name = attachment.filename if attachment and getattr(attachment, 'filename', None) else None
    # Own course check
    try:
        cid = ObjectId(course_id)
    except Exception:
        flash("Invalid course.", "danger")
        return redirect(url_for("teacher_assignments"))
    course = courses.find_one({"_id": cid})
    if not course or str(course.get("instructor_id")) != instructor_id:
        flash("Not authorized for this course.", "danger")
        return redirect(url_for("teacher_assignments"))
    # Prepare attachment saving if provided
    attachment_path = None
    if attachment and attachment_name:
        uploads_dir = os.path.join(app.root_path, 'static', 'uploads', 'assignments')
        os.makedirs(uploads_dir, exist_ok=True)
        safe_name = secure_filename(attachment_name)
        unique_name = f"{ObjectId()}_{safe_name}"
        full_path = os.path.join(uploads_dir, unique_name)
        attachment.save(full_path)
        # Store relative path from static/
        attachment_path = os.path.join('uploads', 'assignments', unique_name)

    assignments.insert_one({
        "course_id": cid,
        "course_code": course.get("code"),
        "course_title": course.get("title"),
        "title": title,
        "description": description,
        "deadline": deadline,
        "attachment_name": attachment_name,
        "attachment_path": attachment_path,
        "created_at": datetime.utcnow(),
    })
    flash("Assignment created.", "success")
    return redirect(url_for("teacher_assignments"))


@app.route("/teacher/assignments/<aid>/update", methods=["POST"]) 
def teacher_assignments_update(aid):
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    try:
        oid = ObjectId(aid)
    except Exception:
        flash("Invalid assignment id.", "danger")
        return redirect(url_for("teacher_assignments"))
    a = assignments.find_one({"_id": oid})
    if not a:
        flash("Assignment not found.", "warning")
        return redirect(url_for("teacher_assignments"))
    course = courses.find_one({"_id": a.get("course_id")})
    if not course or str(course.get("instructor_id")) != instructor_id:
        flash("Not authorized to update this assignment.", "danger")
        return redirect(url_for("teacher_assignments"))
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    deadline = (request.form.get("deadline") or "").strip()
    update_doc = {"title": title, "description": description, "deadline": deadline}
    assignments.update_one({"_id": oid}, {"$set": update_doc})
    flash("Assignment updated.", "success")
    return redirect(url_for("teacher_assignments"))


@app.route("/teacher/assignments/<aid>/delete", methods=["POST"]) 
def teacher_assignments_delete(aid):
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    try:
        oid = ObjectId(aid)
    except Exception:
        flash("Invalid assignment id.", "danger")
        return redirect(url_for("teacher_assignments"))
    a = assignments.find_one({"_id": oid})
    if not a:
        flash("Assignment not found.", "warning")
        return redirect(url_for("teacher_assignments"))
    course = courses.find_one({"_id": a.get("course_id")})
    if not course or str(course.get("instructor_id")) != instructor_id:
        flash("Not authorized to delete this assignment.", "danger")
        return redirect(url_for("teacher_assignments"))
    assignments.delete_one({"_id": oid})
    flash("Assignment deleted.", "success")
    return redirect(url_for("teacher_assignments"))


@app.route("/assignments/<aid>/download")
def assignment_download(aid):
    try:
        oid = ObjectId(aid)
    except Exception:
        flash("Invalid assignment id.", "danger")
        return redirect(url_for("student_assignments"))
    a = assignments.find_one({"_id": oid})
    if not a or not a.get("attachment_path"):
        flash("No attachment available for this assignment.", "warning")
        return redirect(url_for("student_assignments"))
    rel_path = a.get("attachment_path")
    download_name = a.get("attachment_name") or "attachment"
    full_path = os.path.join(app.root_path, 'static', rel_path)
    try:
        return send_file(full_path, as_attachment=True, download_name=download_name)
    except Exception as e:
        flash("Could not download attachment.", "danger")
        return redirect(url_for("student_assignments"))


# -------------------------
# Teacher: Evaluate Submissions
# -------------------------

@app.route("/teacher/submissions")
def teacher_evaluate():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    owned = list(courses.find({"instructor_id": ObjectId(instructor_id)})) if instructor_id else []
    course_ids = [c["_id"] for c in owned]
    # Filter assignments for owned courses
    assign_ids = [a["_id"] for a in assignments.find({"course_id": {"$in": course_ids}})] if course_ids else []
    # Gather submissions
    items = []
    users_map = {}
    if assign_ids:
        for s in submissions.find({"assignment_id": {"$in": [str(x) for x in assign_ids]}}):
            # assignment_id stored as string earlier; normalize
            aid = s.get("assignment_id")
            a = assignments.find_one({"_id": ObjectId(aid)}) if aid and len(aid) == 24 else None
            course = courses.find_one({"_id": a.get("course_id")}) if a else None
            stu_id = s.get("student_id")
            if stu_id and stu_id not in users_map:
                try:
                    udoc = users.find_one({"_id": ObjectId(stu_id)})
                    users_map[stu_id] = udoc
                except Exception:
                    users_map[stu_id] = None
            items.append({
                "_id": str(s.get("_id")),
                "assignment_title": a.get("title") if a else "",
                "course_code": course.get("code") if course else "",
                "student_name": (users_map.get(stu_id) or {}).get("name", ""),
                "filename": s.get("filename"),
                "file_path": s.get("file_path"),
                "score": s.get("score"),
                "total_marks": s.get("total_marks"),
                "feedback": s.get("feedback"),
            })
    return render_template("teacher/evaluate.html", user=session.get("user"), role="Teacher", items=items)


@app.route("/teacher/submissions/<sid>/grade", methods=["POST"]) 
def teacher_grade_submission(sid):
    if not require_teacher():
        return redirect(url_for("login"))
    
    # Check if this is a delete action
    if request.args.get('delete') == 'true':
        try:
            oid = ObjectId(sid)
            # Get the submission to find related records
            sub_doc = submissions.find_one({"_id": oid})
            if sub_doc:
                # Remove grade and feedback from submission
                submissions.update_one({"_id": oid}, {"$unset": {
                    "score": "",
                    "total_marks": "",
                    "feedback": "",
                    "graded_on": ""
                }})
                
                # Also remove from results collection
                if sub_doc.get("assignment_id"):
                    results.delete_one({
                        "student_id": sub_doc.get("student_id"),
                        "component": "Assignment",
                        "ref_id": sub_doc.get("assignment_id")
                    })
                
                flash("Grade and feedback deleted successfully.", "success")
            else:
                flash("Submission not found.", "danger")
        except Exception as e:
            flash(f"Error deleting grade: {str(e)}", "danger")
        return redirect(url_for("teacher_evaluate"))
    
    # Normal save/update action
    score = request.form.get("score")
    total_marks = request.form.get("total_marks")
    feedback = request.form.get("feedback")
    
    try:
        oid = ObjectId(sid)
        # Update submission with grade and feedback
        sub_doc = submissions.find_one({"_id": oid})
        submissions.update_one({"_id": oid}, {"$set": {
            "score": score,
            "total_marks": total_marks,
            "feedback": feedback,
            "graded_on": datetime.utcnow(),
        }})

        # Also upsert into results so it shows under student Results
        if sub_doc:
            assignment_id = sub_doc.get("assignment_id")
            student_id = sub_doc.get("student_id")
            a = None
            course_code = None
            try:
                a = assignments.find_one({"_id": ObjectId(assignment_id)}) if assignment_id and len(assignment_id) == 24 else None
            except Exception:
                a = None
            if a:
                course = courses.find_one({"_id": a.get("course_id")}) if a.get("course_id") else None
                course_code = (course or {}).get("code")

            # Build results doc shape expected by student_results template
            # Using course_code for readability in UI
            results.update_one(
                {
                    "student_id": student_id,
                    "component": "Assignment",
                    "ref_id": assignment_id,
                },
                {
                    "$set": {
                        "course_id": course_code or str((a or {}).get("course_id") or ""),
                        "marks_obtained": score,
                        "total_marks": total_marks,
                        "feedback": feedback,
                        "reference_title": (a or {}).get("title"),
                        "updated_at": datetime.utcnow(),
                    },
                    "$setOnInsert": {"created_at": datetime.utcnow()},
                },
                upsert=True,
            )
        flash("Grade saved.", "success")
    except Exception:
        flash("Could not grade submission.", "danger")
    return redirect(url_for("teacher_evaluate"))


# -------------------------
# Announcements (Teacher + Student)
# -------------------------

@app.route("/teacher/announcements", methods=["GET", "POST"]) 
def teacher_announcements():
    if not require_teacher():
        return redirect(url_for("login"))
    instructor_id = session.get("user_id")
    owned = list(courses.find({"instructor_id": ObjectId(instructor_id)})) if instructor_id else []
    course_ids = [str(c["_id"]) for c in owned]
    message = None
    if request.method == "POST":
        course_id = request.form.get("course_id")
        title = (request.form.get("title") or "").strip()
        content = (request.form.get("content") or "").strip()
        try:
            cid = ObjectId(course_id)
        except Exception:
            flash("Invalid course.", "danger")
            return redirect(url_for("teacher_announcements"))
        course = courses.find_one({"_id": cid})
        if not course or str(course.get("instructor_id")) != instructor_id:
            flash("Not authorized for this course.", "danger")
            return redirect(url_for("teacher_announcements"))
        announcements.insert_one({
            "course_id": cid,
            "title": title,
            "content": content,
            "created_at": datetime.utcnow(),
            "created_by": ObjectId(instructor_id),
        })
        flash("Announcement posted.", "success")
        return redirect(url_for("teacher_announcements"))
    # List teacher announcements
    items = []
    for a in announcements.find({"course_id": {"$in": [ObjectId(cid) for cid in course_ids]}}).sort("created_at", -1):
        course = next((c for c in owned if c["_id"] == a.get("course_id")), None)
        items.append({
            "_id": str(a.get("_id")),
            "course_code": course.get("code") if course else "",
            "title": a.get("title"),
            "content": a.get("content"),
            "created_at": a.get("created_at"),
        })
    # For create form
    simple_courses = [{"_id": str(c["_id"]), "code": c.get("code"), "title": c.get("title")} for c in owned]
    return render_template("teacher/announcements.html", user=session.get("user"), role="Teacher", items=items, courses=simple_courses)


@app.route("/teacher/announcements/<aid>/delete", methods=["POST"]) 
def teacher_announcement_delete(aid):
    if not require_teacher():
        return redirect(url_for("login"))
    try:
        oid = ObjectId(aid)
        announcements.delete_one({"_id": oid})
        flash("Announcement deleted.", "success")
    except Exception:
        flash("Could not delete.", "danger")
    return redirect(url_for("teacher_announcements"))

@app.route("/teacher/enrollments/<eid>/decline", methods=["POST"]) 
def teacher_enrollment_decline(eid):
    if not require_teacher():
        return redirect(url_for("login"))
    try:
        oid = ObjectId(eid)
        enrollments.update_one({"_id": oid}, {"$set": {"status": "declined"}})
        flash("Enrollment declined.", "info")
    except Exception:
        flash("Could not decline enrollment.", "danger")
    return redirect(url_for("teacher_students"))


@app.route("/teacher/enrollments/<eid>/remove", methods=["POST"]) 
def teacher_enrollment_remove(eid):
    if not require_teacher():
        return redirect(url_for("login"))
    try:
        oid = ObjectId(eid)
        # Permanently remove the enrollment so the row disappears from the list
        res = enrollments.delete_one({"_id": oid})
        if res.deleted_count:
            flash("Enrollment removed.", "success")
        else:
            flash("Enrollment not found.", "warning")
    except Exception:
        flash("Could not remove student.", "danger")
    return redirect(url_for("teacher_students"))

# -------------------------
# Admin: Manage Users
# -------------------------

@app.route("/admin/users")
def admin_users():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    items = list(users.find().sort("created_at", -1))
    return render_template("admin/users.html", user=session.get("user"), role="Admin", items=items)


@app.route("/admin/users/add", methods=["POST"]) 
def admin_users_add():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    password = (request.form.get("password") or "").strip()
    role_val = (request.form.get("role") or "").strip() or "Student"
    if not email or not password or not name:
        flash("Name, email and password are required.", "warning")
        return redirect(url_for("admin_users"))
    try:
        users.insert_one({
            "name": name,
            "email": email,
            "password": password,
            "role": role_val,
            "created_at": datetime.utcnow()
        })
        flash("User added.", "success")
    except Exception as e:
        # Likely duplicate email
        flash("Could not add user: " + str(e), "danger")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<uid>/update", methods=["POST"]) 
def admin_users_update(uid):
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    try:
        oid = ObjectId(uid)
    except Exception:
        flash("Invalid user id.", "danger")
        return redirect(url_for("admin_users"))
    update_doc = {}
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    role_val = (request.form.get("role") or "").strip()
    password = (request.form.get("password") or "").strip()
    if name: update_doc["name"] = name
    if email: update_doc["email"] = email
    if role_val: update_doc["role"] = role_val
    if password: update_doc["password"] = password
    if not update_doc:
        flash("No changes provided.", "info")
        return redirect(url_for("admin_users"))
    try:
        users.update_one({"_id": oid}, {"$set": update_doc})
        flash("User updated.", "success")
    except Exception as e:
        flash("Could not update user: " + str(e), "danger")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<uid>/delete", methods=["POST"]) 
def admin_users_delete(uid):
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    try:
        oid = ObjectId(uid)
        res = users.delete_one({"_id": oid})
        if res.deleted_count:
            flash("User deleted.", "success")
        else:
            flash("User not found.", "warning")
    except Exception as e:
        flash("Could not delete user: " + str(e), "danger")
    return redirect(url_for("admin_users"))

# -------------------------
# Student Feature Routes
# -------------------------
@app.route("/student/profile", methods=["GET", "POST"])
def student_profile():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    message = None
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        dob = request.form.get("dob")
        contact = request.form.get("contact")

        update_doc = {}
        if name: update_doc["name"] = name
        if email: update_doc["email"] = email
        if password: update_doc["password"] = password
        if dob: update_doc["dob"] = dob
        if contact: update_doc["contact"] = contact

        if update_doc:
            users.update_one({"email": current_email}, {"$set": update_doc})
            # If email changed, update session anchor
            if email:
                session["email"] = email
            if name:
                session["user"] = name
            message = "Profile updated successfully."

    profile = users.find_one({"email": session.get("email")}, {"password": 0})
    return render_template("student/profile.html", user=session.get("user"), role="Student", profile=profile)


@app.route("/student/courses")
def student_courses():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    student = users.find_one({"email": current_email})
    if not student:
        return redirect(url_for("login"))
    student_id = str(student["_id"])
    # Join enrollments -> courses
    enrolled = []
    for enr in enrollments.find({"user_id": student_id, "status": {"$ne": "dropped"}}):
        course = courses.find_one({"_id": enr.get("course_id")})
        if not course and enr.get("course_code"):
            course = courses.find_one({"code": enr.get("course_code")})
        enrolled.append({
            "enrollment_id": str(enr.get("_id")),
            "course_id": str(enr.get("course_id")) if enr.get("course_id") else None,
            "course_code": course.get("code") if course else enr.get("course_code"),
            "course_title": course.get("title") if course else "(Unknown Course)",
        })
    return render_template("student/courses.html", user=session.get("user"), role="Student", enrolled=enrolled)


@app.route("/student/enroll", methods=["GET", "POST"])
def student_enroll():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    message = None
    # Identify student and current enrollments
    student = users.find_one({"email": current_email})
    student_id = str(student["_id"]) if student else None

    # Helper to compute available (active) courses excluding already enrolled
    def compute_available():
        enrolled_codes = set()
        for enr in enrollments.find({"user_id": student_id, "status": {"$ne": "dropped"}}):
            code = enr.get("course_code")
            if not code and enr.get("course_id"):
                # fallback lookup if needed
                crs = courses.find_one({"_id": enr.get("course_id")})
                code = crs.get("code") if crs else None
            if code:
                enrolled_codes.add(code)
        all_active = list(courses.find({"$or": [{"active": True}, {"active": {"$exists": False}}]}).sort("code", 1))
        return [c for c in all_active if c.get("code") not in enrolled_codes]

    if request.method == "POST":
        code = (request.form.get("course_code") or "").strip().upper()
        if code:
            # Ensure course exists (lightweight create if not)
            course = courses.find_one({"code": code})
            if not course:
                message = f"Course {code} not found. Please select from the list."
                available_courses = compute_available()
                return render_template(
                    "student/enroll.html",
                    user=session.get("user"),
                    role="Student",
                    message=message,
                    available_courses=available_courses,
                )
            # Upsert enrollment
            try:
                enrollments.insert_one({
                    "user_id": student_id,
                    "course_id": course["_id"],
                    "course_code": course.get("code"),
                    "status": "active"
                })
                message = f"Enrolled in {code}."
            except Exception:
                # Likely duplicate due to unique index
                enrollments.update_one(
                    {"user_id": student_id, "course_id": course["_id"]},
                    {"$set": {"status": "active"}}
                )
                message = f"Already enrolled. Status set to active."
        else:
            message = "Please select a course."

    available_courses = compute_available()
    return render_template(
        "student/enroll.html",
        user=session.get("user"),
        role="Student",
        message=message,
        available_courses=available_courses,
    )


@app.route("/student/drop/<course_id>", methods=["POST"]) 
def student_drop_course(course_id):
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    student = users.find_one({"email": current_email})
    if student:
        student_id = str(student["_id"])
        oid = None
        try:
            oid = ObjectId(course_id)
        except Exception:
            oid = None
        query = {"user_id": student_id}
        if oid is not None:
            query["course_id"] = oid
        enrollments.update_many(query, {"$set": {"status": "dropped"}})
    return redirect(url_for("student_courses"))


@app.route("/student/assignments")
def student_assignments():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    student = users.find_one({"email": current_email})
    if not student:
        return redirect(url_for("login"))
    student_id = str(student["_id"])
    # Find active enrollments
    course_ids = [e.get("course_id") for e in enrollments.find({"user_id": student_id, "status": {"$ne": "dropped"}})]
    # Exclude assignments already submitted by this student
    submitted_ids_raw = [s.get("assignment_id") for s in submissions.find({"student_id": student_id}, {"assignment_id": 1, "_id": 0})]
    submitted_oids = []
    for sid in submitted_ids_raw:
        try:
            if sid:
                submitted_oids.append(ObjectId(sid))
        except Exception:
            # Ignore non-ObjectId values just in case
            pass
    query = {"course_id": {"$in": course_ids}} if course_ids else {}
    if submitted_oids:
        query["_id"] = {"$nin": submitted_oids}
    items = list(assignments.find(query)) if course_ids else []
    # Normalize for template: convert ObjectId to str, expose attachment info
    view_items = []
    for a in items:
        c_code = a.get("course_code")
        c_title = a.get("course_title")
        if not c_code or not c_title:
            try:
                cdoc = courses.find_one({"_id": a.get("course_id")})
                if cdoc:
                    c_code = c_code or cdoc.get("code")
                    c_title = c_title or cdoc.get("title")
            except Exception:
                pass
        view_items.append({
            "_id": str(a.get("_id")),
            "course_code": c_code,
            "course_title": c_title,
            "title": a.get("title"),
            "description": a.get("description"),
            "deadline": a.get("deadline"),
            "attachment_name": a.get("attachment_name"),
            "has_attachment": True if a.get("attachment_path") else False,
        })
    return render_template("student/assignments.html", user=session.get("user"), role="Student", assignments=view_items)


@app.route("/student/submissions", methods=["GET", "POST"]) 
def student_submissions():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    message = None
    student = users.find_one({"email": current_email})
    student_id = str(student["_id"]) if student else None
    if request.method == "POST":
        assignment_id = (request.form.get("assignment_id") or "").strip()
        file = request.files.get("file")
        filename = file.filename if file and getattr(file, 'filename', None) else None
        # Optional: basic validation of assignment id shape
        try:
            _ = ObjectId(assignment_id)
        except Exception:
            flash("Invalid assignment id.", "warning")
            return redirect(url_for("student_assignments"))

        # Lookup assignment to capture course info
        a_doc = None
        try:
            a_doc = assignments.find_one({"_id": ObjectId(assignment_id)})
        except Exception:
            a_doc = None
        course_id = (a_doc or {}).get("course_id")
        # Resolve readable course code/title from courses if not present on assignment
        course_code = (a_doc or {}).get("course_code")
        course_title = (a_doc or {}).get("course_title")
        if (not course_code or not course_title) and course_id:
            try:
                cdoc = courses.find_one({"_id": course_id})
                if cdoc:
                    course_code = course_code or cdoc.get("code")
                    course_title = course_title or cdoc.get("title")
            except Exception:
                pass

        file_path = None
        if file and filename:
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads', 'submissions')
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = secure_filename(filename)
            unique_name = f"{ObjectId()}_{safe_name}"
            full_path = os.path.join(uploads_dir, unique_name)
            file.save(full_path)
            # store relative path from static/ using forward slashes for url_for('static', ...)
            file_path = "uploads/submissions/" + unique_name

        # Capture student display info
        stu_name = (student or {}).get("name") or (student or {}).get("email")
        stu_email = (student or {}).get("email")

        submissions.insert_one({
            "assignment_id": assignment_id,
            "student_id": student_id,
            "student_name": stu_name,
            "student_email": stu_email,
            "course_id": course_id,
            "course_code": course_code,
            "course_title": course_title,
            "filename": filename,
            "file_path": file_path,
            "submitted_on": datetime.utcnow(),
        })
        message = "Submission uploaded successfully."
    items = list(submissions.find({"student_id": student_id})) if student_id else []
    return render_template("student/submissions.html", user=session.get("user"), role="Student", submissions_list=items, message=message)


@app.route("/student/results")
def student_results():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    student = users.find_one({"email": current_email})
    student_id = str(student["_id"]) if student else None
    items = list(results.find({"student_id": student_id})) if student_id else []
    return render_template("student/results.html", user=session.get("user"), role="Student", results_list=items)


@app.route("/student/progress")
def student_progress():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    student = users.find_one({"email": current_email})
    student_id = str(student["_id"]) if student else None

    labels = []
    values = []
    colors = []
    if student_id:
        # Aggregate results per course code
        per_course = {}
        for r in results.find({"student_id": student_id}):
            course_code = r.get("course_id")  # stored as readable course code
            mo = r.get("marks_obtained")
            tm = r.get("total_marks")
            pct = None
            try:
                mo_f = float(mo) if mo is not None and mo != "" else None
                tm_f = float(tm) if tm is not None and tm != "" else None
                if mo_f is not None and tm_f and tm_f > 0:
                    pct = (mo_f / tm_f) * 100.0
                elif mo_f is not None and (tm is None or tm == ""):
                    # If total not provided, assume marks_obtained is already a percentage
                    pct = mo_f
            except Exception:
                pct = None
            if course_code and pct is not None:
                per_course.setdefault(course_code, []).append(pct)
        # Average per course
        for course_code, arr in per_course.items():
            if not arr:
                continue
            avg = sum(arr) / len(arr)
            labels.append(course_code)
            values.append(round(avg, 2))
        # Colors by thresholds
        for v in values:
            if v >= 80:
                colors.append('#198754')  # green
            elif v >= 50:
                colors.append('#ffc107')  # yellow
            else:
                colors.append('#dc3545')  # red

    return render_template(
        "student/progress.html",
        user=session.get("user"),
        role="Student",
        chart_labels=labels,
        chart_values=values,
        chart_colors=colors,
    )


@app.route("/student/announcements")
def student_announcements():
    current_email = session.get("email")
    if not current_email:
        return redirect(url_for("login"))
    student = users.find_one({"email": current_email})
    student_id = str(student["_id"]) if student else None
    # Find active enrollments and their course_ids
    course_ids = [e.get("course_id") for e in enrollments.find({"user_id": student_id, "status": {"$ne": "dropped"}})] if student_id else []
    items = []
    if course_ids:
        for a in announcements.find({"course_id": {"$in": course_ids}}).sort("created_at", -1):
            course = courses.find_one({"_id": a.get("course_id")})
            items.append({
                "course_code": course.get("code") if course else "",
                "title": a.get("title"),
                "content": a.get("content"),
                "created_at": a.get("created_at"),
            })
    return render_template("student/announcements.html", user=session.get("user"), role="Student", items=items)


# -------------------------
# Teacher Profile
# -------------------------

@app.route("/teacher/profile", methods=["GET", "POST"]) 
def teacher_profile_page():
    if not require_teacher():
        return redirect(url_for("login"))
    current_email = session.get("email")
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip()
        subject = (request.form.get("subject_specialization") or "").strip()
        update_doc = {}
        if name: update_doc["name"] = name
        if email: update_doc["email"] = email
        if subject: update_doc["subject_specialization"] = subject
        if update_doc:
            users.update_one({"email": current_email}, {"$set": update_doc})
            if email:
                session["email"] = email
            if name:
                session["user"] = name
            flash("Profile updated.", "success")
    profile = users.find_one({"email": session.get("email")}, {"password": 0})
    return render_template("teacher/profile.html", user=session.get("user"), role="Teacher", profile=profile)

@app.route("/student")
def student_dashboard():
    if session.get("role") != "Student":
        return redirect(url_for("home"))
    return render_template("student_dashboard.html", user=session.get("user"), role="Student")

@app.route("/analyst")
def analyst_dashboard():
    if session.get("role") != "Analyst":
        return redirect(url_for("home"))
    return render_template("analyst_dashboard.html", user=session.get("user"), role="Analyst")

# -------------------------
# Analyst: Data Explorer & Analytics
# -------------------------

def require_analyst():
    if session.get("role") != "Analyst":
        return False
    return True

# -------------------------
# Analyst: Model Train & Predict (Linear Regression)
# -------------------------

def _coerce_dropout(val):
    try:
        if isinstance(val, (int, float)):
            return 1 if float(val) >= 1 else 0
        s = str(val or "").strip().lower()
        if s in ("1", "yes", "y", "true", "drop", "dropped", "dropped out"):
            return 1
        if s in ("0", "no", "n", "false", "continue", "continued"):
            return 0
        # fallback
        return int(float(s))
    except Exception:
        return 0

def _prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names to exact expected labels (trim spaces)
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Required columns based on problem statement
    expected = [
        "Student_ID", "Age", "Gender", "Attendance_Percentage", "Assignment_Completion",
        "Test_Score", "Family_Income", "Parental_Education", "Access_to_Resources", "Dropout"
    ]
    # Keep only columns that exist and are expected
    present = [c for c in expected if c in df.columns]
    df = df[present]
    # Basic cleanups
    if "Dropout" in df.columns:
        df["Dropout"] = df["Dropout"].map(_coerce_dropout)
    # Coerce numeric fields
    for num_col in ["Age", "Attendance_Percentage", "Assignment_Completion", "Test_Score", "Family_Income"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    # Normalize categorical fields
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    if "Access_to_Resources" in df.columns:
        df["Access_to_Resources"] = df["Access_to_Resources"].astype(str).str.strip().str.title()
    if "Parental_Education" in df.columns:
        df["Parental_Education"] = df["Parental_Education"].astype(str).str.strip()
    # Drop rows with any NA in features/target
    df = df.dropna(how="any")
    return df

def _build_features(df: pd.DataFrame):
    # Exclude identifiers and target
    feature_cols = [c for c in df.columns if c not in ("Student_ID", "Dropout")]
    X = df[feature_cols].copy()
    # One-hot encode categoricals
    X_enc = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype == "object"], drop_first=True)
    y = df["Dropout"] if "Dropout" in df.columns else None
    return X_enc, y

def _align_input_with_model(input_df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    X_enc = pd.get_dummies(input_df, columns=[c for c in input_df.columns if input_df[c].dtype == "object"], drop_first=True)
    for col in model_cols:
        if col not in X_enc.columns:
            X_enc[col] = 0
    # Keep only model columns and correct order
    X_enc = X_enc[model_cols]
    return X_enc

@app.route("/analyst/model-train-predict")
def analyst_model_train_predict_page():
    if not require_analyst():
        return redirect(url_for("login"))
    return render_template("analyst/model_train_predict.html", user=session.get("user"), role="Analyst")

@app.route("/api/analyst/model/upload", methods=["POST"])
def api_analyst_model_upload():
    if not require_analyst():
        return jsonify({"error": "unauthorized"}), 403
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "CSV file is required with form field 'file'"}), 400
    try:
        file.stream.seek(0)
        df_raw = pd.read_csv(file)
        # Do a light cleanup but do not force specific schema
        df = df_raw.copy()
        # Trim whitespace in column names
        df.columns = [str(c).strip() for c in df.columns]
        # Drop rows that are entirely empty
        df = df.dropna(how="all")
        headers = list(df.columns)
        # Save dataset rows in a separate collection with an ownership tag
        dataset_doc = {
            "analyst": session.get("user"),
            "analyst_email": session.get("email"),
            "headers": headers,
            "created_at": datetime.utcnow(),
        }
        res = ml_datasets.insert_one(dataset_doc)
        dataset_id = str(res.inserted_id)
        # Persist full dataset rows with reference to dataset_id
        # Keep a safe limited sample of rows to avoid very large inserts on huge CSVs
        records = df.head(5000).to_dict(orient="records")
        for r in records:
            r["dataset_id"] = dataset_id
        if records:
            # use batches to avoid oversized payloads
            batch = 1000
            for i in range(0, len(records), batch):
                ml_dataset_rows.insert_many(records[i:i+batch], ordered=False)
        # Store a small sample and count on the parent doc
        sample_rows = df.head(50).to_dict(orient="records")
        ml_datasets.update_one({"_id": res.inserted_id}, {"$set": {"sample": sample_rows, "count": int(len(df))}})
        return jsonify({"dataset_id": dataset_id, "headers": headers, "count": int(len(df))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyst/model/train", methods=["POST"])
def api_analyst_model_train():
    if not require_analyst():
        return jsonify({"error": "unauthorized"}), 403
    file = request.files.get("file")
    try:
        if file:
            file.stream.seek(0)
            df_raw = pd.read_csv(file)
        else:
            return jsonify({"error": "CSV file is required for training"}), 400
        # Prepare generic training pipeline
        df = df_raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if df.shape[1] < 2:
            return jsonify({"error": "CSV must contain at least 2 columns (features + target)"}), 400
        target = (request.form.get("target") or "").strip()
        # allow user to request model type; default to linear for binary classification
        req_model = (request.form.get("model") or "").strip().lower()
        if not target or target not in df.columns:
            target = df.columns[-1]
        # Only drop rows where target is missing; keep feature NaNs to be handled downstream
        df = df[df[target].notna()]
        feature_cols = [c for c in df.columns if c != target]
        if len(feature_cols) == 0:
            return jsonify({"error": "No feature columns found after selecting target"}), 400
        X_df = df[feature_cols].copy()
        y_raw = df[target].copy()
        # Decide task type
        uniques = pd.Series(y_raw).dropna().unique()
        uniq_count = len(uniques)
        # Treat as classification when target is categorical/text OR small discrete set (<=20 classes)
        is_categorical_dtype = (y_raw.dtype == "object" or str(y_raw.dtype).startswith("category"))
        is_classification = (uniq_count >= 2) and (is_categorical_dtype or uniq_count <= 20)
        is_binary = is_classification and uniq_count == 2
        # Build encoders for categorical features
        encoders = {}
        X_enc = X_df.copy()
        for col in X_enc.columns:
            if X_enc[col].dtype == "object" or str(X_enc[col].dtype).startswith("category"):
                le = LabelEncoder()
                try:
                    X_enc[col] = le.fit_transform(X_enc[col].astype(str).fillna(""))
                    encoders[col] = le
                except Exception:
                    X_enc[col] = 0
            else:
                X_enc[col] = pd.to_numeric(X_enc[col], errors="coerce").fillna(0)
        # Prepare y
        if is_classification:
            y_classes = None
            if y_raw.dtype == "object" or str(y_raw.dtype).startswith("category"):
                _ly = LabelEncoder()
                y = _ly.fit_transform(y_raw.astype(str))
                try:
                    y_classes = list(getattr(_ly, "classes_", []))
                except Exception:
                    y_classes = None
            else:
                y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)
                # If numeric but low cardinality, still store sorted unique as classes for mapping
                try:
                    y_classes = sorted([int(v) for v in pd.Series(y).dropna().unique().tolist()]) if uniq_count > 2 else None
                except Exception:
                    y_classes = None
            # Simple leakage guard: drop any feature identical to y (or its inverse)
            drop_cols = []
            y_arr = np.array(y).ravel()
            for col in list(X_enc.columns):
                col_arr = np.array(X_enc[col]).ravel()
                if col_arr.shape == y_arr.shape and (np.array_equal(col_arr, y_arr) or np.array_equal(col_arr, 1 - y_arr)):
                    drop_cols.append(col)
            if drop_cols:
                X_enc = X_enc.drop(columns=drop_cols)

            # Group-balanced sample weights across Access_to_Resources within each class
            sample_weight = None
            try:
                if "Access_to_Resources" in X_df.columns:
                    grp = X_df["Access_to_Resources"].astype(str).str.title().fillna("")
                    import pandas as _pd
                    dfw = _pd.DataFrame({"y": y, "g": grp})
                    counts = dfw.groupby(["y", "g"]).size().rename("n").reset_index()
                    key_to_n = {(int(r["y"]), str(r["g"])): int(r["n"]) for _, r in counts.iterrows()}
                    sw = []
                    for i in range(len(dfw)):
                        k = (int(dfw.at[i, "y"]), str(dfw.at[i, "g"]))
                        n = float(key_to_n.get(k, 1.0))
                        sw.append(1.0 / n)
                    sw = np.array(sw, dtype=float)
                    m = float(np.mean(sw)) if np.isfinite(np.mean(sw)) else 1.0
                    if m == 0: m = 1.0
                    sw = sw / m
                    sample_weight = sw
            except Exception:
                sample_weight = None
            # For multiclass, default to RF; for binary, allow logistic default
            use_rf = (req_model == "rf") or (not is_binary)
            if use_rf:
                # Apply more conservative RF to reduce overfitting
                clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced",
                )
                # 5-fold Stratified CV for realistic validation
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = []
                for tr_idx, va_idx in skf.split(X_enc.values, y):
                    if sample_weight is not None:
                        clf.fit(X_enc.values[tr_idx], np.array(y)[tr_idx], sample_weight=sample_weight[tr_idx])
                    else:
                        clf.fit(X_enc.values[tr_idx], np.array(y)[tr_idx])
                    va_pred = clf.predict(X_enc.values[va_idx])
                    try:
                        score = float(balanced_accuracy_score(np.array(y)[va_idx], va_pred))
                    except Exception:
                        score = None
                    if score is not None:
                        cv_scores.append(score)
                val_acc = float(np.mean(cv_scores)) if cv_scores else None
                # Fit final model on all data for persistence
                if sample_weight is not None:
                    clf.fit(X_enc.values, y, sample_weight=sample_weight)
                else:
                    clf.fit(X_enc.values, y)
                model_doc = {
                    "type": "random_forest",
                    "analyst": session.get("user"),
                    "analyst_email": session.get("email"),
                    "created_at": datetime.utcnow(),
                    "feature_columns": list(X_enc.columns),
                    "model_blob": Binary(pickle.dumps(clf)),
                    "encoders_blob": Binary(pickle.dumps(encoders)) if encoders else None,
                    "target": target,
                    "val_accuracy": val_acc,
                    "is_binary": bool(is_binary),
                    "y_classes": y_classes if (y_classes and len(y_classes) >= 2) else None,
                }
                models.insert_one(model_doc)
                try:
                    admin_notifs_col.insert_one({
                        "type": "model_trained",
                        "analyst": session.get("user"),
                        "analyst_email": session.get("email"),
                        "rows_used": int(len(df)),
                        "model_type": "random_forest",
                        "val_accuracy": val_acc,
                        "created_at": datetime.utcnow(),
                        "message": f"Model trained by {session.get('user')} ({session.get('email')})"
                    })
                except Exception:
                    pass
                label_counts = {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()}
                total_labels = sum(label_counts.values()) or 1
                label_ratio = {k: round(v/total_labels, 4) for k, v in label_counts.items()}
                return jsonify({
                    "message": "RandomForest model trained",
                    "feature_columns": list(X_enc.columns),
                    "rows_used": int(len(df)),
                    "label_counts": label_counts,
                    "label_ratio": label_ratio,
                    "val_accuracy": val_acc,
                    "target": target,
                    "target_classes": y_classes if (y_classes and len(y_classes) >= 2) else None,
                })
            else:
                # Default: Logistic Regression (linear) for binary classification
                from sklearn.linear_model import LogisticRegression as _LogReg
                # Stronger regularization to reduce overfitting
                logreg = _LogReg(max_iter=1000, class_weight="balanced", C=0.1, solver="liblinear")
                # 5-fold Stratified CV
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = []
                for tr_idx, va_idx in skf.split(X_enc.values, y):
                    if sample_weight is not None:
                        logreg.fit(X_enc.values[tr_idx], np.array(y)[tr_idx], sample_weight=sample_weight[tr_idx])
                    else:
                        logreg.fit(X_enc.values[tr_idx], np.array(y)[tr_idx])
                    va_pred = logreg.predict(X_enc.values[va_idx])
                    try:
                        score = float(balanced_accuracy_score(np.array(y)[va_idx], va_pred))
                    except Exception:
                        score = None
                    if score is not None:
                        cv_scores.append(score)
                val_acc = float(np.mean(cv_scores)) if cv_scores else None
                # Fit final model on all data for persistence
                if sample_weight is not None:
                    logreg.fit(X_enc.values, y, sample_weight=sample_weight)
                else:
                    logreg.fit(X_enc.values, y)
                model_doc = {
                    "type": "logistic_regression",
                    "analyst": session.get("user"),
                    "analyst_email": session.get("email"),
                    "created_at": datetime.utcnow(),
                    "feature_columns": list(X_enc.columns),
                    # store linear params for portable prediction
                    "coef": list(logreg.coef_.ravel().tolist()),
                    "intercept": float(logreg.intercept_.ravel()[0]) if hasattr(logreg.intercept_, "ravel") else float(np.array(logreg.intercept_).ravel()[0]),
                    "encoders_blob": Binary(pickle.dumps(encoders)) if encoders else None,
                    "target": target,
                    "val_accuracy": val_acc,
                    "is_binary": True,
                }
                models.insert_one(model_doc)
                try:
                    admin_notifs_col.insert_one({
                        "type": "model_trained",
                        "analyst": session.get("user"),
                        "analyst_email": session.get("email"),
                        "rows_used": int(len(df)),
                        "model_type": "logistic_regression",
                        "val_accuracy": val_acc,
                        "created_at": datetime.utcnow(),
                        "message": f"Model trained by {session.get('user')} ({session.get('email')})"
                    })
                except Exception:
                    pass
                label_counts = {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()}
                total_labels = sum(label_counts.values()) or 1
                label_ratio = {k: round(v/total_labels, 4) for k, v in label_counts.items()}
                return jsonify({
                    "message": "Logistic Regression model trained",
                    "feature_columns": list(X_enc.columns),
                    "rows_used": int(len(df)),
                    "label_counts": label_counts,
                    "label_ratio": label_ratio,
                    "val_accuracy": val_acc,
                    "target": target,
                })
        else:
            # Regression fallback
            y = pd.to_numeric(y_raw, errors="coerce").fillna(0)
            lin = LinearRegression()
            # 5-fold CV with R^2
            try:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                Xv = X_enc.values
                yv = y.values
                for tr_idx, va_idx in kf.split(Xv):
                    lin.fit(Xv[tr_idx], yv[tr_idx])
                    y_hat = lin.predict(Xv[va_idx])
                    try:
                        s = float(r2_score(yv[va_idx], y_hat))
                    except Exception:
                        s = None
                    if s is not None and np.isfinite(s):
                        scores.append(s)
                val_accuracy = float(np.mean(scores)) if scores else None
            except Exception:
                val_accuracy = None
            # Fit final model on all data
            lin.fit(X_enc.values, y.values)
            model_doc = {
                "type": "linear_regression",
                "analyst": session.get("user"),
                "analyst_email": session.get("email"),
                "created_at": datetime.utcnow(),
                "feature_columns": list(X_enc.columns),
                "coef": list(lin.coef_.ravel().tolist()) if hasattr(lin.coef_, "ravel") else list(np.array(lin.coef_).tolist()),
                "intercept": float(lin.intercept_),
                "encoders_blob": Binary(pickle.dumps(encoders)) if encoders else None,
                "target": target,
                "val_accuracy": val_accuracy,
                "is_binary": False,
            }
            models.insert_one(model_doc)
            # Notify admins
            try:
                admin_notifs_col.insert_one({
                    "type": "model_trained",
                    "analyst": session.get("user"),
                    "analyst_email": session.get("email"),
                    "rows_used": int(len(df)),
                    "model_type": "linear_regression",
                    "val_accuracy": val_accuracy,
                    "created_at": datetime.utcnow(),
                    "message": f"Model trained by {session.get('user')} ({session.get('email')})"
                })
            except Exception:
                pass
            # Prepare encoder class listing for UI
            try:
                encoder_classes = {k: list(getattr(v, 'classes_', [])) for k, v in (encoders or {}).items()}
            except Exception:
                encoder_classes = {}
            return jsonify({
                "message": "Linear Regression model trained",
                "feature_columns": list(X_enc.columns),
                "rows_used": int(len(df)),
                "val_accuracy": val_accuracy,
                "target": target,
                "encoder_classes": encoder_classes,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyst/model/predict", methods=["POST"])
def api_analyst_model_predict():
    if not require_analyst():
        return jsonify({"error": "unauthorized"}), 403
    try:
        payload = request.json or {}
        # Fetch latest model for this analyst
        mdl = models.find_one({"analyst_email": session.get("email")}, sort=[("created_at", -1)])
        if not mdl:
            return jsonify({"error": "No trained model found. Train the model first."}), 400
        model_cols = mdl.get("feature_columns", [])
        # Build a 1-row DataFrame directly from provided payload
        # Only keep keys that belong to model feature columns
        row = {k: payload.get(k) for k in model_cols if k in payload}
        df_in = pd.DataFrame([row])
        # Coerce numerics
        for nc in df_in.columns:
            # we'll coerce later for RF; for linear we scale/align below
            pass
        # For RandomForest models, apply saved label encoders and skip scaling
        mtype = mdl.get("type")
        if mtype == "random_forest" and mdl.get("encoders_blob") is not None:
            try:
                encoders = pickle.loads(mdl["encoders_blob"]) or {}
            except Exception:
                encoders = {}
            # apply encoders deterministically; unseen values -> 0
            for cat_col, le in encoders.items():
                if cat_col in df_in.columns:
                    val = str(df_in.at[0, cat_col]) if cat_col in df_in.columns else ""
                    classes = list(getattr(le, "classes_", []))
                    try:
                        idx = classes.index(val)
                    except ValueError:
                        idx = 0
                    df_in[cat_col] = idx
            # ensure numeric types for the rest
            for nc in df_in.columns:
                if nc not in encoders:
                    df_in[nc] = pd.to_numeric(df_in[nc], errors="coerce").fillna(0)
            # align to RF feature order without one-hot
            for col in model_cols:
                if col not in df_in.columns:
                    df_in[col] = 0
            X_enc = df_in[model_cols]
        elif mtype in ("linear_regression", "logistic_regression"):
            # Generic linear regression over original feature columns (encoders optional)
            try:
                encoders = pickle.loads(mdl.get("encoders_blob") or b"") or {}
            except Exception:
                encoders = {}
            # ensure all model columns exist
            for col in model_cols:
                if col not in df_in.columns:
                    df_in[col] = 0
            # apply encoders if present, otherwise numeric coercion
            for col in model_cols:
                if col in encoders:
                    val = str(df_in.at[0, col]) if col in df_in.columns else ""
                    classes = list(getattr(encoders[col], "classes_", []))
                    try:
                        idx = classes.index(val)
                    except ValueError:
                        idx = 0
                    df_in[col] = idx
                else:
                    df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0)
            X_enc = df_in[model_cols]
        else:
            # Legacy models: apply scaling and one-hot
            mdl_mins = (mdl.get("numeric_mins") or {})
            mdl_maxs = (mdl.get("numeric_maxs") or {})
            for c, vmin in mdl_mins.items():
                vmax = float(mdl_maxs.get(c, vmin + 1.0))
                denom = (vmax - float(vmin)) if (vmax - float(vmin)) != 0 else 1.0
                if c in df_in.columns:
                    df_in[c] = (pd.to_numeric(df_in[c], errors="coerce") - float(vmin)) / denom
            X_enc = _align_input_with_model(df_in, model_cols)
        # Prepare debug snapshot of feature vector
        debug_vec = {}
        try:
            debug_vec = {c: (float(df_in[c].iloc[0]) if c in df_in.columns else None) for c in model_cols}
        except Exception:
            debug_vec = {}
        if mtype == "random_forest" and mdl.get("model_blob") is not None:
            try:
                rf = pickle.loads(mdl["model_blob"])
                proba = rf.predict_proba(X_enc.values)[0]
                y_pred = rf.predict(X_enc.values)[0]
                # Determine probability of predicted class (or max prob for multiclass)
                try:
                    prob = float(np.max(proba))
                except Exception:
                    prob = None
                # Map numeric class back to original label if available
                y_classes = mdl.get("y_classes") or []
                try:
                    if isinstance(y_pred, (int, np.integer)) and y_classes and int(y_pred) < len(y_classes):
                        label = str(y_classes[int(y_pred)])
                    else:
                        label = (str(y_pred) if not isinstance(y_pred, (int, np.integer)) else int(y_pred))
                except Exception:
                    label = str(y_pred)
            except Exception:
                prob = 0.5
                label = "Unknown"
        else:
            # Linear-family models stored as coef/intercept
            coef = np.array(mdl.get("coef", []), dtype=float).ravel()
            intercept = float(mdl.get("intercept", 0.0))
            linear = float(np.dot(X_enc.values[0], coef) + intercept)
            if mtype == "logistic_regression":
                prob = 1.0 / (1.0 + np.exp(-linear))
                label = 1 if prob >= 0.5 else 0
            elif mtype == "linear_regression" and bool(mdl.get("is_binary")):
                # Enforce binary mapping for regression model trained on binary target
                prob = 1.0 / (1.0 + np.exp(-linear))
                label = 1 if prob >= 0.5 else 0
            else:
                prob = None
                label = str(round(linear, 4))
        # Build charts from last uploaded sample for key metrics
        last_ds = ml_datasets.find_one({"analyst_email": session.get("email")}, sort=[("created_at", -1)]) or {}
        sample = last_ds.get("sample", [])
        charts = {}
        try:
            if sample:
                def hist(values, bins):
                    buckets = {b:0 for b in bins}
                    for v in values:
                        try:
                            x = float(v)
                        except Exception:
                            x = 0.0
                        if x < 20: buckets["0-20"] += 1
                        elif x < 40: buckets["20-40"] += 1
                        elif x < 60: buckets["40-60"] += 1
                        elif x < 80: buckets["60-80"] += 1
                        else: buckets["80-100"] += 1
                    return buckets
                bins = ["0-20","20-40","40-60","60-80","80-100"]
                att_vals = [r.get("Attendance_Percentage", 0) for r in sample]
                asg_vals = [r.get("Assignment_Completion", 0) for r in sample]
                tst_vals = [r.get("Test_Score", 0) for r in sample]
                att_hist = hist(att_vals, bins)
                asg_hist = hist(asg_vals, bins)
                tst_hist = hist(tst_vals, bins)
                charts["attendance_hist"] = {"labels":bins,"data":[att_hist[b] for b in bins]}
                charts["assignment_hist"] = {"labels":bins,"data":[asg_hist[b] for b in bins]}
                charts["test_hist"] = {"labels":bins,"data":[tst_hist[b] for b in bins]}
                # Means for pie chart
                import numpy as _np
                def _safe_mean(arr):
                    try:
                        vals = _np.array([float(x) for x in arr if x is not None])
                        return float(_np.nanmean(vals)) if vals.size else 0.0
                    except Exception:
                        return 0.0
                charts["means"] = {
                    "labels": ["Attendance %", "Assignment Completion", "Test Score"],
                    "data": [
                        round(_safe_mean(att_vals), 2),
                        round(_safe_mean(asg_vals), 2),
                        round(_safe_mean(tst_vals), 2)
                    ]
                }
        except Exception:
            charts.setdefault("attendance_hist", {"type":"bar","labels":[],"data":[]})
            charts.setdefault("assignment_hist", {"type":"bar","labels":[],"data":[]})
            charts.setdefault("test_hist", {"type":"bar","labels":[],"data":[]})
        # Only round probability if numeric
        try:
            prob_out = round(prob, 3) if isinstance(prob, (int, float)) else None
        except Exception:
            prob_out = None
        # Preserve binary 0/1 as ints; otherwise round numeric predictions for display stability
        is_bin_flag = bool(mdl.get("is_binary"))
        try:
            if isinstance(label, (int, float)) and (int(round(float(label))) in (0,1)) and is_bin_flag:
                label_out = int(round(float(label)))
            elif isinstance(label, (int, float)):
                label_out = round(float(label), 2)
            else:
                label_out = label
        except Exception:
            label_out = label
        return jsonify({
            "prediction": label_out,
            "probability": prob_out,
            "charts": charts,
            "used_features": model_cols,
            "input_vector": debug_vec,
            "target": mdl.get("target"),
            "is_binary": is_bin_flag,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyst/model/save", methods=["POST"])
def api_analyst_model_save():
    if not require_analyst():
        return jsonify({"error": "unauthorized"}), 403
    try:
        payload = request.json or {}
        pred = payload.get("prediction")
        charts = payload.get("charts")
        inputs = payload.get("inputs")
        probability = payload.get("probability")
        # Round numeric fields
        try:
            pred_out = round(float(pred), 2)
        except Exception:
            pred_out = pred
        try:
            prob_out = round(float(probability), 3)
        except Exception:
            prob_out = probability
        target = payload.get("target")
        if not target:
            try:
                mdl = models.find_one({"analyst_email": session.get("email")}, sort=[("created_at", -1)]) or {}
                target = mdl.get("target")
            except Exception:
                target = None
        if not pred:
            return jsonify({"error": "prediction is required"}), 400
        try:
            ih = hashlib.sha256(json.dumps(inputs or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        except Exception:
            ih = None
        now = datetime.utcnow()
        recent_window = now - timedelta(minutes=10)
        q = {
            "analyst_email": session.get("email"),
            "source": "auto",
            "inputs_hash": ih,
            "created_at": {"$gte": recent_window}
        }
        try:
            existing = ml_predictions.find_one(q, sort=[("created_at", -1)])
        except Exception:
            existing = None
        if existing and existing.get("_id"):
            try:
                ml_predictions.update_one(
                    {"_id": existing.get("_id")},
                    {"$set": {
                        "prediction": pred_out,
                        "probability": prob_out,
                        "target": target,
                        "charts": charts,
                        "inputs": inputs,
                        "source": "manual",
                    }}
                )
            except Exception:
                pass
        else:
            ml_predictions.insert_one({
                "type": "ml",
                "analyst": session.get("user"),
                "analyst_email": session.get("email"),
                "inputs": inputs,
                "prediction": pred_out,
                "probability": prob_out,
                "target": target,
                "charts": charts,
                "source": "manual",
                "inputs_hash": ih,
                "created_at": now,
            })
        return jsonify({"message": "Saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_DATASET_COLLECTIONS = {
    "attendance": "attendance",
    "grades": "results",
    "enrollments": "enrollments",
    "demographics": "demographics",
    "lms": "lms_events",
    "academic_records": "academic_records",
}

@app.route("/analyst/datasets")
def analyst_datasets():
    if not require_analyst():
        return redirect(url_for("login"))
    # Provide counts per dataset for quick overview
    counts = {}
    try:
        for key, coll_name in _DATASET_COLLECTIONS.items():
            counts[key] = mongo.db[coll_name].count_documents({})
    except Exception:
        counts = {}
    return render_template("analyst/datasets.html", user=session.get("user"), role="Analyst", counts=counts)

@app.route("/api/analyst/dataset")
def api_analyst_dataset():
    if not require_analyst():
        return jsonify({"error": "unauthorized"}), 403
    name = (request.args.get("name") or "").strip()
    if name not in _DATASET_COLLECTIONS:
        return jsonify({"error": f"Unknown dataset: {name}"}), 400
    coll = mongo.db[_DATASET_COLLECTIONS[name]]
    q = (request.args.get("q") or "").strip()
    limit = min(int(request.args.get("limit", 500) or 500), 2000)
    sort = (request.args.get("sort") or "").strip()
    order = (request.args.get("order") or "asc").lower()
    try:
        cursor = coll.find({})
        if q:
            # Basic text search across string fields (client-side like behavior)
            cursor = coll.find({"$or": [
                {k: {"$regex": q, "$options": "i"}} for k in coll.find_one() or {} if isinstance((coll.find_one() or {}).get(k), str)
            ]})
        if sort:
            cursor = cursor.sort(sort, 1 if order != "desc" else -1)
        rows = list(cursor.limit(limit))
        # Convert ObjectId to string and prepare headers
        clean_rows = []
        headers = set()
        def fmt_dt(val):
            try:
                if isinstance(val, dict) and "$date" in val:
                    s = val.get("$date")
                    try:
                        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return str(s)
                if isinstance(val, datetime):
                    return val.strftime("%Y-%m-%d %H:%M:%S")
                return val
            except Exception:
                return val
        def clean_value(v):
            # Normalize None/NaN/inf and textual equivalents to empty string; format date-like objects
            try:
                if v is None:
                    return ""
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return ""
                if isinstance(v, dict) and "$date" in v:
                    return fmt_dt(v)
                if isinstance(v, datetime):
                    return fmt_dt(v)
                if isinstance(v, str) and v.strip().lower() in ("nan", "none", "null", "inf", "-inf"):
                    return ""
                return v
            except Exception:
                return v
        # Precompile forbidden character regex and define columns to exclude from symbol checks
        comp_pat = re.compile(FORBIDDEN_CHAR_PATTERN)
        # Exclude fields where certain symbols are expected/allowed (dates, contacts, feedback text, etc.)
        exclude_cols = {"term", "dob", "email", "phone", "address", "event_time", "details", "date",
                        "feedback", "remarks", "remark", "comment", "comments"}
        # Always drop rows that contain a literal '?' anywhere (even in excluded columns)
        hard_pat = re.compile(r"\?")

        # Dataset-specific rule: for 'grades', allow incomplete rows (optional fields may be empty)
        require_complete = (name != "grades")

        for r in rows:
            cr = {}
            for k, v in r.items():
                if k == "_id":
                    cr[k] = str(v)
                else:
                    cr[k] = clean_value(v)
                headers.add(k)
            # Skip rows that have forbidden special symbols in checked columns
            has_forbidden = False
            try:
                for ck, cv in cr.items():
                    if ck == "_id":
                        continue
                    # If any column has a literal '?', drop the row immediately
                    if isinstance(cv, str) and hard_pat.search(cv):
                        has_forbidden = True
                        break
                    # Otherwise check forbidden pattern only for non-excluded columns
                    if ck in exclude_cols:
                        continue
                    if isinstance(cv, str) and comp_pat.search(cv):
                        has_forbidden = True
                        break
            except Exception:
                has_forbidden = has_forbidden or False
            if has_forbidden:
                continue

            # Skip rows based on completeness only if required
            if require_complete:
                non_id_keys = [k for k in cr.keys() if k != "_id"]
                if non_id_keys and all((str(cr[k]).strip() != "") for k in non_id_keys):
                    clean_rows.append(cr)
            else:
                clean_rows.append(cr)
        headers = [h for h in sorted(headers)]
        return jsonify({"name": name, "headers": headers, "rows": clean_rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyst/visualizations")
def analyst_visualizations():
    if not require_analyst():
        return redirect(url_for("login"))
    return render_template("analyst/visualizations.html", user=session.get("user"), role="Analyst")

@app.route("/api/analyst/charts/overview")
def api_analyst_charts_overview():
    if not require_analyst():
        return jsonify({"error": "unauthorized"}), 403
    # Build simple aggregates for charts
    data = {}
    try:
        # Course demand: enrollments per course_code
        pipeline = [
            {"$group": {"_id": {"$ifNull": ["$course_code", "$course_id"]}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        demand = list(enrollments.aggregate(pipeline))
        data["course_demand"] = {
            "labels": [str(d.get("_id") or "Unknown") for d in demand],
            "values": [int(d.get("count") or 0) for d in demand]
        }
    except Exception:
        data["course_demand"] = {"labels": [], "values": []}
    try:
        # Attendance patterns: present vs absent counts
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        att = list(attendance.aggregate(pipeline))
        data["attendance_status"] = {
            "labels": [str(d.get("_id") or "Unknown") for d in att],
            "values": [int(d.get("count") or 0) for d in att]
        }
    except Exception:
        data["attendance_status"] = {"labels": [], "values": []}
    try:
        # Student performance trends: average score by course or term if available
        pipeline = [
            {"$group": {"_id": {"course": "$course_id"}, "avg_score": {"$avg": "$score"}}},
            {"$sort": {"avg_score": -1}},
            {"$limit": 10}
        ]
        perf = list(results.aggregate(pipeline))
        data["performance"] = {
            "labels": [str((d.get("_id") or {}).get("course") or "Unknown") for d in perf],
            "values": [float(d.get("avg_score") or 0) for d in perf]
        }
    except Exception:
        data["performance"] = {"labels": [], "values": []}
    try:
        # Dropout proxy: count of LMS events of type 'drop' if exists
        pipeline = [
            {"$match": {"event_type": {"$regex": "drop", "$options": "i"}}},
            {"$group": {"_id": "$course_code", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        drops = list(lms_events.aggregate(pipeline))
        data["dropout"] = {
            "labels": [str(d.get("_id") or "Unknown") for d in drops],
            "values": [int(d.get("count") or 0) for d in drops]
        }
    except Exception:
        data["dropout"] = {"labels": [], "values": []}
    return jsonify(data)

@app.route("/analyst/predictions")
def analyst_predictions():
    if not require_analyst():
        return redirect(url_for("login"))
    # Compute simple heuristic predictions if no ML outputs exist
    at_risk_students = []
    high_low_courses = {"high": [], "low": []}
    upcoming_perf = []
    course_marks = {}
    course_labels = {}

    # Helpers to resolve display labels
    # New: resolve to a COURSE CODE only; return None if cannot resolve
    def resolve_course_code(raw):
        try:
            if not raw:
                return None
            s = str(raw).strip()
            if not s:
                return None
            # If looks like a 24-hex ObjectId, try DB lookup to extract code
            if len(s) == 24:
                try:
                    int(s, 16)
                    c = courses.find_one({"_id": ObjectId(s)})
                    code = (c or {}).get("code")
                    return (code or None)
                except Exception:
                    pass
            # Try direct lookups by id/code/title/name to fetch code
            c = (courses.find_one({"_id": s}) or
                 courses.find_one({"code": s}) or
                 courses.find_one({"title": s}) or
                 courses.find_one({"name": s}))
            if c and c.get("code"):
                return c.get("code")
            # Otherwise, if it's not a 24-hex string, treat as already a code-like value
            try:
                int(s, 16)
                # if it gets here and is hex, it's an unresolved id -> drop
                return None
            except Exception:
                return s
        except Exception:
            return None
    def resolve_course_display(raw):
        try:
            if not raw:
                return "UNKNOWN"
            s = str(raw).strip()
            disp = None
            # Try ObjectId lookup
            try:
                if len(s) == 24:
                    obj = ObjectId(s)
                    c = courses.find_one({"_id": obj})
                    if c:
                        disp = c.get("code") or c.get("title") or c.get("name") or s
            except Exception:
                pass
            # Try raw string id or code/title/name
            if not disp:
                c = (courses.find_one({"_id": s}) or
                     courses.find_one({"code": s}) or
                     courses.find_one({"title": s}) or
                     courses.find_one({"name": s}))
                if c:
                    disp = c.get("code") or c.get("title") or c.get("name") or s
            return (disp or s).strip()
        except Exception:
            return str(raw)

    def resolve_student_display(sid):
        try:
            if not sid:
                return "Unknown"
            s = str(sid)
            d = demographics.find_one({"$or": [{"student_id": s}, {"studentId": s}, {"email": s}]}) or {}
            name = d.get("name") or d.get("full_name") or d.get("email")
            if name:
                return name
            # users by _id
            try:
                if len(s) == 24:
                    u = users.find_one({"_id": ObjectId(s)})
                    if u:
                        return u.get("name") or u.get("email") or s
            except Exception:
                pass
            u = users.find_one({"email": s})
            if u:
                return u.get("name") or u.get("email") or s
            return s
        except Exception:
            return str(sid)

    def prettify_student_label(sid, disp):
        try:
            label = (disp or "").strip()
            if not label:
                label = str(sid)
            # If still looks like a 24-hex ObjectId, shorten
            s = str(label)
            if len(s) == 24:
                try:
                    int(s, 16)
                    return f"ID …{s[-6:]}"
                except Exception:
                    pass
            return label
        except Exception:
            return str(disp or sid)

    try:
        # Risk: low attendance or low average score
        # Build attendance rate per student
        att_pipeline = [
            {"$group": {"_id": "$student_id", "present": {"$sum": {"$cond": [{"$eq": ["$status", "present"]}, 1, 0]}}, "total": {"$sum": 1}}}
        ]
        att_rates = {str(d["_id"]): (d.get("present", 0) / max(1, d.get("total", 1))) for d in attendance.aggregate(att_pipeline)}
        # Average scores per student
        score_pipeline = [
            {"$group": {"_id": "$student_id", "avg": {"$avg": "$score"}}}
        ]
        scores = {str(d["_id"]): float(d.get("avg", 0)) for d in results.aggregate(score_pipeline)}
        # Join minimal info from demographics
        for sid, rate in att_rates.items():
            avg = scores.get(sid, None)
            risk = (rate < 0.75) or (avg is not None and avg < 60)
            if risk:
                demo = demographics.find_one({"$or": [{"student_id": sid}, {"studentId": sid}]}) or {}
                at_risk_students.append({
                    "student_id": sid,
                    "attendance_rate": round(rate * 100, 1),
                    "avg_score": round(avg, 1) if avg is not None else None,
                    "name": demo.get("name") or demo.get("full_name") or demo.get("email") or "Unknown"
                })
        at_risk_students = sorted(at_risk_students, key=lambda x: (x.get("avg_score") or 0))[:50]
    except Exception:
        at_risk_students = []
    try:
        # High/low enrollment courses: resolve each enrollment to a course CODE only; drop unresolved
        counts = {}
        for e in enrollments.find({}, {"course_id":1, "course_code":1, "status":1}):
            if e.get("status") == "dropped":
                continue
            raw_course = e.get("course_code") or e.get("course_id")
            code = resolve_course_code(raw_course)
            if not code:
                continue
            key = str(code).strip().upper()
            counts[key] = counts.get(key, 0) + 1
        items = sorted(({"course": k, "count": v} for k, v in counts.items()), key=lambda x: x["count"], reverse=True)
        # Threshold-based split: Low = count <= 1, High = count > 1
        high = [it for it in items if (it.get("count", 0) or 0) > 1][:10]
        low_candidates = sorted(items, key=lambda x: x["count"])  # ascending for lows
        low = [it for it in low_candidates if (it.get("count", 0) or 0) <= 1][:10]
        high_low_courses = {"high": high, "low": low}
    except Exception:
        high_low_courses = {"high": [], "low": []}
    try:
        # Upcoming performance proxy: compute per-course average percentage from submissions
        def to_float(v):
            try:
                if v is None:
                    return 0.0
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v).strip().replace('%','')
                return float(s) if s else 0.0
            except Exception:
                return 0.0

        accum_course = {}
        for doc in submissions.find({}, {"course_code":1, "course_id":1, "course":1, "course_title":1, "code":1, "title":1, "name":1, "score":1, "total_marks":1}):
            raw_course = (doc.get("course_code") or doc.get("course_id") or doc.get("course") or doc.get("course_title") or
                          doc.get("code") or doc.get("title") or doc.get("name"))
            course_code = resolve_course_code(raw_course)
            score = to_float(doc.get("score"))
            total = to_float(doc.get("total_marks"))
            pct = (score/total*100.0) if total > 0 else score
            if pct > 0 and course_code:
                accum_course.setdefault(course_code, []).append(pct)
        items = []
        for c, arr in accum_course.items():
            if not arr:
                continue
            items.append({"course": c, "predicted_avg": round(sum(arr)/max(1, len(arr)), 1)})
        # Fallback to results if submissions had no data
        if not items:
            pipe = [
                {"$group": {"_id": {"course": {"$ifNull": ["$course_code", "$course_id"]}}, "avg_score": {"$avg": "$score"}}},
                {"$sort": {"avg_score": -1}},
                {"$limit": 20}
            ]
            tmp = list(results.aggregate(pipe))
            items = []
            for d in tmp:
                raw = ((d.get("_id") or {}).get("course"))
                code = resolve_course_code(raw)
                if not code:
                    continue
                items.append({
                    "course": code,
                    "predicted_avg": round(float(d.get("avg_score") or 0), 1)
                })
        upcoming_perf = sorted(items, key=lambda x: x.get("predicted_avg", 0), reverse=True)[:20]
    except Exception:
        upcoming_perf = []
    # Build per-course student average marks for interactive charts
    try:
        def to_float(v):
            try:
                if v is None:
                    return 0.0
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v).strip().replace('%','')
                return float(s) if s else 0.0
            except Exception:
                return 0.0

        # Collect raw values per (course_code -> student -> [values]) from submissions
        accum = {}
        display_overrides = {}
        try:
            for doc in submissions.find({}, {"course_code":1, "course_id":1, "course":1, "course_title":1, "code":1, "title":1, "name":1, "student_id":1, "studentId":1, "user_id":1, "userId":1, "student":1, "student_email":1, "email":1, "student_name":1, "score":1, "total_marks":1}):
                raw_course = (doc.get("course_code") or doc.get("course_id") or doc.get("course") or doc.get("course_title") or
                              doc.get("code") or doc.get("title") or doc.get("name"))
                course_code = resolve_course_code(raw_course)
                if not course_code:
                    continue
                course_id = str(course_code).strip().upper()
                course_labels[course_id] = course_code
                sid = (doc.get("student_id") or doc.get("studentId") or doc.get("user_id") or
                       doc.get("userId") or doc.get("student") or doc.get("email") or doc.get("student_email") or
                       doc.get("student_name") or doc.get("name"))
                sid = str(sid or "Unknown")
                # Save a nicer display name if provided
                disp_hint = (doc.get("student_name") or doc.get("name") or doc.get("email") or doc.get("student_email"))
                if disp_hint:
                    display_overrides.setdefault(course_id, {})[sid] = str(disp_hint)
                score = to_float(doc.get("score"))
                total = to_float(doc.get("total_marks"))
                pct = (score/total*100.0) if total > 0 else score
                if course_id and sid and pct > 0:
                    accum.setdefault(course_id, {}).setdefault(sid, []).append(pct)
        except Exception:
            pass

        # Also collect from results as percentages (assume score already in 0..100)
        try:
            for doc in results.find({}, {"course_id":1, "course_code":1, "course":1, "course_title":1, "student_id":1, "user_id":1, "student_email":1, "email":1, "name":1, "score":1}):
                raw_course = (doc.get("course_code") or doc.get("course_id") or doc.get("course") or doc.get("course_title"))
                course_code = resolve_course_code(raw_course)
                if not course_code:
                    continue
                course_id = str(course_code).strip().upper()
                if course_id not in course_labels:
                    course_labels[course_id] = course_code
                sid = (doc.get("student_id") or doc.get("user_id") or doc.get("student_email") or doc.get("email") or doc.get("name") or "Unknown")
                sid = str(sid)
                disp_hint = (doc.get("name") or doc.get("email") or doc.get("student_email"))
                if disp_hint:
                    display_overrides.setdefault(course_id, {})[sid] = str(disp_hint)
                pct = to_float(doc.get("score"))
                if course_id and sid and pct > 0:
                    accum.setdefault(course_id, {}).setdefault(sid, []).append(pct)
        except Exception:
            pass

        # Average per student and prepare labels/values
        for course_id, by_student in accum.items():
            labeled = []
            for sid, arr in by_student.items():
                if not arr:
                    continue
                val = sum(arr) / max(1, len(arr))
                # Prefer collected display hints per course, else resolve via DB lookups; then prettify
                raw_disp = (display_overrides.get(course_id, {}).get(sid)) or resolve_student_display(sid)
                disp = prettify_student_label(sid, raw_disp)
                labeled.append({"label": disp, "value": round(val, 1)})
            items_sorted = sorted(labeled, key=lambda x: x.get("value", 0), reverse=True)[:20]
            if items_sorted:
                course_marks[course_id] = {
                    "labels": [it["label"] for it in items_sorted],
                    "values": [it["value"] for it in items_sorted],
                }
        # Ensure course_labels only contain resolvable course codes; drop any UNKNOWN
        for cid in list(course_marks.keys()):
            lbl = (course_labels.get(cid) or "").strip()
            if not lbl:
                # drop entries with no valid course code
                course_marks.pop(cid, None)
                course_labels.pop(cid, None)
    except Exception:
        course_marks = {}
    return render_template("analyst/predictions.html", user=session.get("user"), role="Analyst", at_risk=at_risk_students, high_low=high_low_courses, upcoming=upcoming_perf, course_marks=course_marks, course_labels=course_labels)

@app.route("/analyst/reports")
def analyst_reports():
    if not require_analyst():
        return redirect(url_for("login"))
    items = []
    selected_target = (request.args.get("target") or "").strip()
    # Treat '(Unlabeled)' or empty as no filter
    if selected_target in ("", "(Unlabeled)"):
        selected_target = ""
    try:
        email = session.get("email")
        base_q = {"type": "ml", "analyst_email": email}
        q = dict(base_q)
        if selected_target:
            q["target"] = selected_target
        cur = ml_predictions.find(q).sort("created_at", -1).limit(500)
        items = [{
            "id": str(doc.get("_id")),
            "prediction": doc.get("prediction"),
            "probability": doc.get("probability"),
            "created_at": doc.get("created_at"),
            "inputs": doc.get("inputs") or {},
            "target": doc.get("target"),
        } for doc in cur]
        # Distinct targets with counts for navigation
        try:
            agg = [
                {"$match": base_q},
                {"$group": {"_id": {"target": "$target"}, "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            tg = list(ml_predictions.aggregate(agg))
            # Only keep labeled targets (non-empty strings), excluding literal '(Unlabeled)' variants
            targets = []
            for d in tg:
                tval = ((d.get("_id") or {}).get("target") or "").strip()
                if not tval:
                    continue
                if tval in ("(Unlabeled)", "Unlabeled", "unlabeled", "UNLABELED"):
                    continue
                targets.append({"target": tval, "count": int(d.get("count") or 0)})
        except Exception:
            targets = []
    except Exception:
        items = []
        targets = []
    # Build dynamic columns from saved inputs across the FILTERED items only
    col_set = []
    try:
        seen = set()
        for it in items:
            for k in (it.get("inputs") or {}).keys():
                if not isinstance(k, str):
                    continue
                if len(k) > 64:
                    continue
                if k.lower() in ("charts", "_id"):
                    continue
                if k not in seen:
                    seen.add(k)
                    col_set.append(k)
        columns = col_set[:10]
    except Exception:
        columns = []

    # If no specific target selected, prepare grouped view per target (each with its own columns)
    grouped = {}
    columns_per = {}
    if not selected_target:
        try:
            # Fetch recent items per target
            for t in targets:
                tgt = t.get("target")
                qg = dict(base_q)
                if tgt:
                    qg["target"] = tgt
                curg = ml_predictions.find(qg).sort("created_at", -1).limit(100)
                arr = [{
                    "id": str(doc.get("_id")),
                    "prediction": doc.get("prediction"),
                    "probability": doc.get("probability"),
                    "created_at": doc.get("created_at"),
                    "inputs": doc.get("inputs") or {},
                    "target": doc.get("target"),
                } for doc in curg]
                if not arr:
                    continue
                grouped[tgt] = arr
                # Columns per target
                seen_t = set()
                cols_t = []
                for it in arr:
                    for k in (it.get("inputs") or {}).keys():
                        if not isinstance(k, str):
                            continue
                        if len(k) > 64:
                            continue
                        if k.lower() in ("charts", "_id"):
                            continue
                        if k not in seen_t:
                            seen_t.add(k)
                            cols_t.append(k)
                columns_per[tgt] = cols_t[:10]
        except Exception:
            grouped = {}
            columns_per = {}

    # Fallback: if grouped empty, provide columns_all derived from 'items' (which includes all predictions for the analyst)
    columns_all = []
    if not grouped:
        try:
            seen_all = set()
            for it in items:
                for k in (it.get("inputs") or {}).keys():
                    if not isinstance(k, str):
                        continue
                    if len(k) > 64:
                        continue
                    if k.lower() in ("charts", "_id"):
                        continue
                    if k not in seen_all:
                        seen_all.add(k)
                        columns_all.append(k)
            columns_all = columns_all[:10]
        except Exception:
            columns_all = []

    return render_template(
        "analyst/reports.html",
        user=session.get("user"),
        role="Analyst",
        items=items,
        columns=columns,
        targets=targets,
        selected_target=selected_target,
        grouped=grouped,
        columns_per=columns_per,
        columns_all=columns_all,
    )

@app.route("/analyst/reports/delete/<pid>", methods=["POST"])
def analyst_reports_delete(pid):
    if not require_analyst():
        return redirect(url_for("login"))
    email = session.get("email")
    try:
        oid = ObjectId(pid)
        ml_predictions.delete_one({"_id": oid, "analyst_email": email})
    except Exception:
        pass
    return redirect(url_for("analyst_reports"))

@app.route("/analyst/manual-predict", methods=["GET", "POST"])
def analyst_manual_predict():
    if not require_analyst():
        return redirect(url_for("login"))
    ctx = {"user": session.get("user"), "role": "Analyst", "input": None, "prediction": None, "error": None}
    if request.method == "POST":
        try:
            student_name = (request.form.get("student_name", "").strip())
            assignment_marks = float(request.form.get("assignment_marks", "").strip())
            test_marks = float(request.form.get("test_marks", "").strip())
            percentage = float(request.form.get("percentage", "").strip())
            attendance = float(request.form.get("attendance", "").strip())
        except Exception:
            ctx["error"] = "Please enter valid numeric values."
            ctx["input"] = {
                "student_name": (request.form.get("student_name", "").strip()),
                "assignment_marks": request.form.get("assignment_marks"),
                "test_marks": request.form.get("test_marks"),
                "percentage": request.form.get("percentage"),
                "attendance": request.form.get("attendance"),
            }
            return render_template("analyst/manual_predict.html", **ctx)
        if not student_name:
            ctx["error"] = "Student name is required."
            ctx["input"] = {
                "student_name": "",
                "assignment_marks": assignment_marks,
                "test_marks": test_marks,
                "percentage": percentage,
                "attendance": attendance,
            }
            return render_template("analyst/manual_predict.html", **ctx)
        dropped = (percentage < 45.0) or (attendance < 40.0) or (assignment_marks < 40.0) or (test_marks < 40.0)
        ctx["input"] = {
            "student_name": student_name,
            "assignment_marks": assignment_marks,
            "test_marks": test_marks,
            "percentage": percentage,
            "attendance": attendance,
        }
        ctx["prediction"] = "Dropped Out" if dropped else "Continue"
        try:
            manual_predictions.insert_one({
                "type": "manual",
                "analyst": session.get("user"),
                "analyst_email": session.get("email"),
                "inputs": ctx.get("input"),
                "prediction": ctx.get("prediction"),
                "created_at": datetime.utcnow(),
            })
        except Exception:
            pass
        return render_template("analyst/manual_predict.html", **ctx)
    return render_template("analyst/manual_predict.html", **ctx)

# ... (rest of the code remains the same)


@app.route("/ingest/csv/<dataset>", methods=["POST"])
def ingest_csv(dataset):
    if dataset not in SUPPORTED_DATASETS:
        return jsonify({"error": f"Unsupported dataset: {dataset}"}), 400
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "CSV file is required with form field 'file'"}), 400
    try:
        records_iter = list(read_csv_stream(file))
        summary = process_records(dataset, records_iter, mongo.db)
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Admin Ingestion UI
# -------------------------

@app.route("/admin/ingestion")
def admin_ingestion():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    return render_template("admin/ingestion.html", user=session.get("user"), role="Admin")

@app.route("/admin/notifications")
def admin_notifications():
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    items = []
    try:
        cur = mongo.db.admin_notifications.find({}).sort("created_at", -1).limit(200)
        for doc in cur:
            items.append({
                "id": str(doc.get("_id")),
                "type": doc.get("type"),
                "analyst": doc.get("analyst"),
                "analyst_email": doc.get("analyst_email"),
                "rows_used": doc.get("rows_used"),
                "model_type": doc.get("model_type"),
                "val_accuracy": doc.get("val_accuracy"),
                "created_at": doc.get("created_at"),
                "message": doc.get("message"),
            })
    except Exception:
        items = []
    return render_template("admin/notifications.html", user=session.get("user"), role="Admin", items=items)


@app.route("/admin/notifications/delete/<nid>", methods=["POST"])
def admin_notifications_delete(nid):
    if session.get("role") != "Admin":
        return redirect(url_for("login"))
    try:
        oid = ObjectId(nid)
        mongo.db.admin_notifications.delete_one({"_id": oid})
    except Exception:
        pass
    return redirect(url_for("admin_notifications"))


# -------------------------
# CSV Cleaning + Import with pandas
# -------------------------

@app.route("/ingest/csv_clean/<dataset>", methods=["POST"])
def ingest_csv_clean(dataset):
    if dataset not in SUPPORTED_DATASETS:
        return jsonify({"error": f"Unsupported dataset: {dataset}"}), 400
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "CSV file is required with form field 'file'"}), 400
    download = (request.form.get("download") or "").lower() in ("1", "true", "yes", "on")
    try:
        # Clean with pandas
        df, clean_summary = clean_with_pandas(dataset, file)
        # Insert cleaned records to Mongo
        df_clean = df.fillna("")
        records = df_clean.to_dict(orient="records")
        insert_summary = process_records(dataset, records, mongo.db)
        # If download requested, return cleaned CSV file
        if download:
            csv_buf = io.StringIO()
            df_clean.to_csv(csv_buf, index=False)
            bytes_buf = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
            return send_file(bytes_buf, mimetype="text/csv", as_attachment=True, download_name=f"{dataset}_cleaned.csv")
        # Otherwise return combined JSON summary
        return jsonify({
            "dataset": dataset,
            "cleaning": clean_summary,
            "insertion": insert_summary,
            "data": {
                "headers": list(df_clean.columns),
                "rows": records
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


 

# Logout
@app.route("/logout")
def logout():
    session.clear()   # sab session data clear kar dega
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
