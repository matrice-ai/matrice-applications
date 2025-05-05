"""
analytics.py - Analytics and Reporting Module

This module generates reports and visualizations from face recognition data.
It analyzes customer visit patterns, staff activity, and system performance.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

from db_manager import db_manager
import utils

# Initialize logger
logger = logging.getLogger('face_recognition.analytics')

def load_visit_data():
    """Load visit data from database"""
    visit_logs = db_manager.load_visit_logs()
    return visit_logs.get("visits", [])

def generate_traffic_report(visits, days=30, output_dir="reports"):
    """
    Generate traffic report showing visit patterns
    
    Args:
        visits: List of visit log entries
        days: Number of days to include
        output_dir: Directory to save reports
    
    Returns:
        Dictionary with report data
    """
    # Filter visits by date
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    
    filtered_visits = []
    for visit in visits:
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            if timestamp >= cutoff:
                filtered_visits.append(visit)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Calculate hourly traffic
    hourly_counts = utils.generate_hourly_traffic_report(filtered_visits, days)
    
    # Create traffic heatmap
    heatmap_data = utils.generate_visit_heatmap(filtered_visits, days)
    
    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Hourly traffic chart
    traffic_fig = utils.plot_hourly_traffic(hourly_counts)
    traffic_fig.savefig(os.path.join(output_dir, "hourly_traffic.png"))
    plt.close(traffic_fig)
    
    # Traffic heatmap
    heatmap_fig = utils.plot_visit_heatmap(heatmap_data)
    heatmap_fig.savefig(os.path.join(output_dir, "traffic_heatmap.png"))
    plt.close(heatmap_fig)
    
    # Summarize data for return
    total_visits = len(filtered_visits)
    
    # Count unique visitors
    unique_customers = set()
    unique_staff = set()
    
    for visit in filtered_visits:
        person_type = visit.get("type")
        person_id = visit.get("personId")
        
        if person_type == "customer":
            unique_customers.add(person_id)
        elif person_type == "staff":
            unique_staff.add(person_id)
    
    # Calculate busiest hour
    busiest_hour = max(hourly_counts.items(), key=lambda x: x[1])
    
    # Calculate busiest day of week
    day_totals = heatmap_data.sum(axis=1)
    busiest_day_idx = np.argmax(day_totals)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    busiest_day = days[busiest_day_idx]
    
    report_data = {
        "time_period": f"Last {days} days",
        "total_visits": total_visits,
        "unique_customers": len(unique_customers),
        "unique_staff": len(unique_staff),
        "busiest_hour": f"{busiest_hour[0]:02d}:00",
        "busiest_hour_count": busiest_hour[1],
        "busiest_day": busiest_day,
        "hourly_data": hourly_counts,
        "heatmap_data": heatmap_data.tolist(),
        "report_time": datetime.now().isoformat()
    }
    
    # Save report data as JSON
    with open(os.path.join(output_dir, "traffic_report.json"), 'w') as f:
        json.dump(report_data, f, indent=4)
    
    return report_data

def generate_customer_report(visits, days=30, output_dir="reports"):
    """
    Generate customer analysis report
    
    Args:
        visits: List of visit log entries
        days: Number of days to include
        output_dir: Directory to save reports
    
    Returns:
        Dictionary with report data
    """
    # Filter visits by date and type
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    
    customer_visits = []
    for visit in visits:
        if visit.get("type") != "customer":
            continue
            
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            if timestamp >= cutoff:
                customer_visits.append(visit)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Get customer database for additional info
    customer_db = db_manager.load_customer_db()
    customers = customer_db.get("customers", [])
    
    # Calculate visit frequency statistics
    visit_stats = utils.get_customer_visit_frequency(visits, days)
    
    # Calculate average dwell time
    avg_dwell_time = utils.calculate_dwell_time(customer_visits)
    
    # Identify returning vs. new customers
    visit_counts = {}
    for visit in customer_visits:
        person_id = visit.get("personId")
        if person_id not in visit_counts:
            visit_counts[person_id] = 0
        visit_counts[person_id] += 1
    
    # Count customers by visit count
    visit_buckets = {
        "1_visit": 0,      # First-time
        "2_visits": 0,     # Returning once
        "3_5_visits": 0,   # Regular
        "6_plus_visits": 0 # Frequent
    }
    
    for _, count in visit_counts.items():
        if count == 1:
            visit_buckets["1_visit"] += 1
        elif count == 2:
            visit_buckets["2_visits"] += 1
        elif 3 <= count <= 5:
            visit_buckets["3_5_visits"] += 1
        else:
            visit_buckets["6_plus_visits"] += 1
    
    # Create customer type pie chart
    plt.figure(figsize=(10, 6))
    labels = ['First-time', 'Returning once', 'Regular', 'Frequent']
    sizes = [
        visit_buckets["1_visit"],
        visit_buckets["2_visits"],
        visit_buckets["3_5_visits"],
        visit_buckets["6_plus_visits"]
    ]
    
    if sum(sizes) > 0:  # Only create chart if we have data
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Customer Visit Frequency')
        plt.savefig(os.path.join(output_dir, "customer_types.png"))
    
    plt.close()
    
    # Collect customer visit trends over time if we have enough data
    if len(customer_visits) >= 10:
        # Group visits by day
        visit_dates = {}
        for visit in customer_visits:
            try:
                timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
                date_str = timestamp.strftime("%Y-%m-%d")
                
                if date_str not in visit_dates:
                    visit_dates[date_str] = 0
                    
                visit_dates[date_str] += 1
            except Exception as e:
                logger.error(f"Error parsing timestamp: {e}")
        
        # Sort dates and create trend chart
        sorted_dates = sorted(visit_dates.keys())
        visit_counts = [visit_dates[date] for date in sorted_dates]
        
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_dates, visit_counts, marker='o')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Customer Visits')
        plt.title('Customer Visit Trend')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "customer_trend.png"))
        plt.close()
    
    # Compile report data
    report_data = {
        "time_period": f"Last {days} days",
        "total_customer_visits": len(customer_visits),
        "unique_customers": len(visit_counts),
        "avg_visits_per_customer": visit_stats["avg_visits_per_customer"],
        "return_rate": visit_stats["return_rate"],
        "avg_dwell_time_seconds": avg_dwell_time,
        "avg_dwell_time_formatted": utils.format_time_delta(avg_dwell_time),
        "customer_types": {
            "first_time": visit_buckets["1_visit"],
            "returning_once": visit_buckets["2_visits"],
            "regular": visit_buckets["3_5_visits"],
            "frequent": visit_buckets["6_plus_visits"]
        },
        "report_time": datetime.now().isoformat()
    }
    
    # Save report data as JSON
    with open(os.path.join(output_dir, "customer_report.json"), 'w') as f:
        json.dump(report_data, f, indent=4)
    
    return report_data

def generate_staff_report(visits, days=30, output_dir="reports"):
    """
    Generate staff activity report
    
    Args:
        visits: List of visit log entries
        days: Number of days to include
        output_dir: Directory to save reports
    
    Returns:
        Dictionary with report data
    """
    # Filter visits by date and type
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    
    staff_visits = []
    for visit in visits:
        if visit.get("type") != "staff":
            continue
            
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            if timestamp >= cutoff:
                staff_visits.append(visit)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Get staff database for names
    staff_db = db_manager.load_staff_db()
    staff_members = {s["staffId"]: s for s in staff_db.get("staff", [])}
    
    # Count visits by staff member
    staff_activity = {}
    for visit in staff_visits:
        staff_id = visit.get("personId")
        if staff_id not in staff_activity:
            staff_activity[staff_id] = {
                "visits": 0,
                "total_confidence": 0,
                "timestamps": []
            }
        
        staff_activity[staff_id]["visits"] += 1
        staff_activity[staff_id]["total_confidence"] += float(visit.get("confidence", 0))
        
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            staff_activity[staff_id]["timestamps"].append(timestamp)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Calculate average confidence and add names
    staff_summary = []
    for staff_id, data in staff_activity.items():
        avg_confidence = data["total_confidence"] / data["visits"] if data["visits"] > 0 else 0
        
        staff_info = staff_members.get(staff_id, {})
        first_name = staff_info.get("firstName", "Unknown")
        last_name = staff_info.get("lastName", "Staff")
        position = staff_info.get("position", "Unknown")
        
        # Calculate frequency (days with visits / total days)
        visit_days = set()
        for ts in data["timestamps"]:
            visit_days.add(ts.date())
        
        # Calculate coverage (percentage of days in period with visits)
        total_days = (now - cutoff).days
        coverage = len(visit_days) / total_days if total_days > 0 else 0
        
        staff_summary.append({
            "staff_id": staff_id,
            "name": f"{first_name} {last_name}",
            "position": position,
            "visits": data["visits"],
            "avg_confidence": avg_confidence,
            "days_present": len(visit_days),
            "coverage": coverage
        })
    
    # Sort by number of visits
    staff_summary.sort(key=lambda x: x["visits"], reverse=True)
    
    # Create staff activity bar chart (top 10 most active)
    top_staff = staff_summary[:10]
    if top_staff:
        plt.figure(figsize=(12, 6))
        names = [f"{s['name'][:12]}..." if len(s['name']) > 15 else s['name'] for s in top_staff]
        visits = [s["visits"] for s in top_staff]
        
        plt.bar(names, visits)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Staff Member')
        plt.ylabel('Recognition Count')
        plt.title('Staff Recognition Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "staff_activity.png"))
        plt.close()
    
    # Compile report data
    report_data = {
        "time_period": f"Last {days} days",
        "total_staff_recognitions": len(staff_visits),
        "unique_staff_detected": len(staff_activity),
        "staff_summary": staff_summary,
        "report_time": datetime.now().isoformat()
    }
    
    # Save report data as JSON
    with open(os.path.join(output_dir, "staff_report.json"), 'w') as f:
        json.dump(report_data, f, indent=4)
    
    return report_data

def generate_system_report(visits, days=30, output_dir="reports"):
    """
    Generate system performance report
    
    Args:
        visits: List of visit log entries
        days: Number of days to include
        output_dir: Directory to save reports
    
    Returns:
        Dictionary with report data
    """
    # Load logs if possible (simplified - would need more structured logging)
    log_file = 'face_recognition_data/face_recognition.log'
    
    log_entries = []
    error_count = 0
    warning_count = 0
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'ERROR' in line:
                        error_count += 1
                    elif 'WARNING' in line:
                        warning_count += 1
                    log_entries.append(line)
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
    
    # Filter visits by date
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    
    filtered_visits = []
    for visit in visits:
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            if timestamp >= cutoff:
                filtered_visits.append(visit)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Calculate recognition accuracy (based on confidence scores)
    confidence_scores = [float(v.get("confidence", 0)) for v in filtered_visits]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # Count visits by location
    location_counts = {}
    for visit in filtered_visits:
        location = visit.get("location", "Unknown")
        if location not in location_counts:
            location_counts[location] = 0
        location_counts[location] += 1
    
    # Plot location distribution if we have location data
    if location_counts:
        plt.figure(figsize=(10, 6))
        locations = list(location_counts.keys())
        counts = [location_counts[loc] for loc in locations]
        
        plt.bar(locations, counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Location')
        plt.ylabel('Recognition Count')
        plt.title('Recognition by Location')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "location_distribution.png"))
        plt.close()
    
    # Compile report data
    report_data = {
        "time_period": f"Last {days} days",
        "total_recognitions": len(filtered_visits),
        "avg_confidence_score": avg_confidence,
        "error_count": error_count,
        "warning_count": warning_count,
        "locations": location_counts,
        "report_time": datetime.now().isoformat()
    }
    
    # Save report data as JSON
    with open(os.path.join(output_dir, "system_report.json"), 'w') as f:
        json.dump(report_data, f, indent=4)
    
    return report_data

def generate_summary_report(traffic_report, customer_report, staff_report, system_report, output_dir="reports"):
    """
    Generate a combined summary report
    
    Args:
        traffic_report: Traffic report data
        customer_report: Customer report data
        staff_report: Staff report data
        system_report: System report data
        output_dir: Directory to save report
    
    Returns:
        None
    """
    days = traffic_report.get("time_period", "").split()[-2]
    
    # Create a simple HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition System Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .report-section {{ margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; width: 45%; }}
            .stat {{ margin: 10px 0; }}
            .stat-label {{ font-weight: bold; }}
            .stat-value {{ color: #0066cc; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Face Recognition System - Summary Report</h1>
        <p>Period: Last {days} days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="container">
            <div class="report-section">
                <h2>Traffic Overview</h2>
                <div class="stat">
                    <span class="stat-label">Total Visits:</span>
                    <span class="stat-value">{traffic_report.get('total_visits', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Unique Customers:</span>
                    <span class="stat-value">{traffic_report.get('unique_customers', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Unique Staff:</span>
                    <span class="stat-value">{traffic_report.get('unique_staff', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Busiest Hour:</span>
                    <span class="stat-value">{traffic_report.get('busiest_hour', 'N/A')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Busiest Day:</span>
                    <span class="stat-value">{traffic_report.get('busiest_day', 'N/A')}</span>
                </div>
                <img src="hourly_traffic.png" alt="Hourly Traffic Chart">
            </div>
            
            <div class="report-section">
                <h2>Customer Analysis</h2>
                <div class="stat">
                    <span class="stat-label">Total Customer Visits:</span>
                    <span class="stat-value">{customer_report.get('total_customer_visits', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Unique Customers:</span>
                    <span class="stat-value">{customer_report.get('unique_customers', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Average Visits per Customer:</span>
                    <span class="stat-value">{customer_report.get('avg_visits_per_customer', 0):.2f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Return Rate:</span>
                    <span class="stat-value">{customer_report.get('return_rate', 0):.2%}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Average Dwell Time:</span>
                    <span class="stat-value">{customer_report.get('avg_dwell_time_formatted', 'N/A')}</span>
                </div>
                <img src="customer_types.png" alt="Customer Types">
            </div>
            
            <div class="report-section">
                <h2>Staff Activity</h2>
                <div class="stat">
                    <span class="stat-label">Total Staff Recognitions:</span>
                    <span class="stat-value">{staff_report.get('total_staff_recognitions', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Unique Staff Detected:</span>
                    <span class="stat-value">{staff_report.get('unique_staff_detected', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Most Active Staff:</span>
                    <span class="stat-value">{staff_report.get('staff_summary', [{}])[0].get('name', 'N/A') if staff_report.get('staff_summary', []) else 'N/A'}</span>
                </div>
                <img src="staff_activity.png" alt="Staff Activity">
            </div>
            
            <div class="report-section">
                <h2>System Performance</h2>
                <div class="stat">
                    <span class="stat-label">Total Recognitions:</span>
                    <span class="stat-value">{system_report.get('total_recognitions', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Average Confidence Score:</span>
                    <span class="stat-value">{system_report.get('avg_confidence_score', 0):.2%}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Error Count:</span>
                    <span class="stat-value">{system_report.get('error_count', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Warning Count:</span>
                    <span class="stat-value">{system_report.get('warning_count', 0)}</span>
                </div>
                <img src="location_distribution.png" alt="Location Distribution">
            </div>
        </div>
        
        <div style="margin-top: 20px; text-align: center; color: #666;">
            <p>Generated by Face Recognition Analytics System</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, "summary_report.html"), 'w') as f:
        f.write(html)
    
    print(f"Summary report saved to {os.path.join(output_dir, 'summary_report.html')}")

def generate_reports(days=30, output_dir="reports"):
    """
    Generate all reports
    
    Args:
        days: Number of days to include
        output_dir: Directory to save reports
    
    Returns:
        None
    """
    print(f"Generating reports for the last {days} days...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load visit data
    visits = load_visit_data()
    
    # Generate individual reports
    traffic_report = generate_traffic_report(visits, days, output_dir)
    customer_report = generate_customer_report(visits, days, output_dir)
    staff_report = generate_staff_report(visits, days, output_dir)
    system_report = generate_system_report(visits, days, output_dir)
    
    # Generate summary report
    generate_summary_report(
        traffic_report,
        customer_report,
        staff_report,
        system_report,
        output_dir
    )
    
    print(f"Report generation complete. Reports saved to {output_dir}/")
    
    return {
        "traffic_report": traffic_report,
        "customer_report": customer_report,
        "staff_report": staff_report,
        "system_report": system_report
    }

if __name__ == "__main__":
    generate_reports() 