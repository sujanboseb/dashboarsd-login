from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from pymongo import MongoClient
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Loaded from .env

# MongoDB connection
raw_password = "sawq#@21"
encoded_password = quote_plus(raw_password)
uri = f"mongodb+srv://sujanboseplant04:{encoded_password}@cluster0.ea3ecul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client["investment"]
users_collection = db["users"]
collection = db["sales"]

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phone = request.form["phone"]
        password = request.form["password"]
        user = users_collection.find_one({"phone": phone, "password": password})
        
        if user:
            session["user"] = phone
            session["role"] = user["role"]
            
            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("user_dashboard"))
        else:
            flash("Invalid credentials. Please try again.", "danger")
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        phone = request.form["phone"]
        password = request.form["password"]
        role = request.form["role"]
        
        if len(phone) != 10 or not phone.isdigit():
            flash("Phone number must be 10 digits.", "danger")
            return redirect(url_for("signup"))
        
        if users_collection.find_one({"phone": phone}):
            flash("Phone number already registered.", "danger")
            return redirect(url_for("signup"))
        
        users_collection.insert_one({"phone": phone, "password": password, "role": role})
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))
    
    return render_template("signup.html")

@app.route("/admin-dashboard")
def admin_dashboard():
    if "user" not in session or session["role"] != "admin":
        flash("Unauthorized access!", "danger")
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")

@app.route("/api/data")
def get_data():
    # Get all product-wise sales data
    data = list(collection.find({}, {"_id": 0}))  # Remove _id for cleaner JSON
    return jsonify(data)

@app.route("/api/monthly-data")
def get_monthly_data():
    # MongoDB aggregation to group by month and calculate total sales per month
    pipeline = [
        {
            "$project": {
                "month": { "$month": "$Order Date" },  # Group by month of Order Date
                "year": { "$year": "$Order Date" },   # Group by year of Order Date
                "Amount": 1
            }
        },
        {
            "$group": {
                "_id": { "year": "$year", "month": "$month" },
                "totalAmount": { "$sum": "$Amount" }
            }
        },
        {
            "$sort": { "_id": 1 }  # Sort by year and month
        }
    ]
    
    data = list(collection.aggregate(pipeline))  # Get the aggregated data

    # Format the data to a more usable structure for the frontend
    months = []
    total_sales = []
    for item in data:
        month = datetime(item["_id"]["year"], item["_id"]["month"], 1).strftime("%b %Y")
        months.append(month)
        total_sales.append(item["totalAmount"])

    return jsonify({"months": months, "total_sales": total_sales})

@app.route("/user-dashboard")
def user_dashboard():
    if "user" not in session or session["role"] != "user":
        flash("Unauthorized access!", "danger")
        return redirect(url_for("login"))
    return render_template("user_dashboard.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
