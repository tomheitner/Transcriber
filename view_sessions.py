#!/usr/bin/env python3
"""
Utility script to view and manage saved live transcription sessions.
"""

import os
import json
import glob
from datetime import datetime

def list_sessions(sessions_dir="live_sessions"):
    """List all saved sessions."""
    if not os.path.exists(sessions_dir):
        print(f"Directory '{sessions_dir}' not found.")
        return []
    
    json_files = glob.glob(os.path.join(sessions_dir, "live_session_*.json"))
    sessions = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                sessions.append({
                    'file': json_file,
                    'session_id': session_data['session_id'],
                    'start_time': session_data['start_time'],
                    'duration': session_data['duration_seconds'],
                    'transcriptions': session_data['total_transcriptions']
                })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return sorted(sessions, key=lambda x: x['start_time'], reverse=True)

def view_session(session_id, sessions_dir="live_sessions"):
    """View a specific session."""
    json_file = os.path.join(sessions_dir, f"live_session_{session_id}.json")
    
    if not os.path.exists(json_file):
        print(f"Session {session_id} not found.")
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"SESSION: {session_data['session_id']}")
        print(f"{'='*60}")
        print(f"Start Time: {session_data['start_time']}")
        print(f"End Time: {session_data['end_time']}")
        print(f"Duration: {session_data['duration_seconds']:.2f} seconds")
        print(f"Total Transcriptions: {session_data['total_transcriptions']}")
        print(f"{'='*60}")
        
        for i, trans in enumerate(session_data['transcriptions'], 1):
            print(f"[{i:03d}] {trans['timestamp']} - {trans['text']}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error reading session: {e}")

def export_session(session_id, output_format="txt", sessions_dir="live_sessions"):
    """Export a session to different formats."""
    json_file = os.path.join(sessions_dir, f"live_session_{session_id}.json")
    
    if not os.path.exists(json_file):
        print(f"Session {session_id} not found.")
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        if output_format.lower() == "txt":
            output_file = os.path.join(sessions_dir, f"export_{session_id}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Live Transcription Session Export\n")
                f.write(f"Session ID: {session_data['session_id']}\n")
                f.write(f"Start Time: {session_data['start_time']}\n")
                f.write(f"End Time: {session_data['end_time']}\n")
                f.write(f"Duration: {session_data['duration_seconds']:.2f} seconds\n")
                f.write(f"Total Transcriptions: {session_data['total_transcriptions']}\n")
                f.write("="*50 + "\n\n")
                
                for i, trans in enumerate(session_data['transcriptions'], 1):
                    f.write(f"[{i:03d}] {trans['timestamp']} - {trans['text']}\n")
            
            print(f"Session exported to: {output_file}")
        
        elif output_format.lower() == "json":
            output_file = os.path.join(sessions_dir, f"export_{session_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"Session exported to: {output_file}")
        
        else:
            print(f"Unsupported format: {output_format}")
    
    except Exception as e:
        print(f"Error exporting session: {e}")

def main():
    """Main function to manage sessions."""
    print("📁 LIVE TRANSCRIPTION SESSION MANAGER")
    print("="*50)
    
    sessions_dir = "live_sessions"
    
    while True:
        print("\nOptions:")
        print("1. List all sessions")
        print("2. View a specific session")
        print("3. Export a session")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            sessions = list_sessions(sessions_dir)
            if sessions:
                print(f"\nFound {len(sessions)} sessions:")
                print("-" * 80)
                for session in sessions:
                    start_time = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                    print(f"ID: {session['session_id']}")
                    print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Duration: {session['duration']:.2f}s")
                    print(f"  Transcriptions: {session['transcriptions']}")
                    print()
            else:
                print("No sessions found.")
        
        elif choice == "2":
            session_id = input("Enter session ID (e.g., 20240101_143022): ").strip()
            view_session(session_id, sessions_dir)
        
        elif choice == "3":
            session_id = input("Enter session ID (e.g., 20240101_143022): ").strip()
            format_choice = input("Export format (txt/json): ").strip().lower()
            if format_choice in ["txt", "json"]:
                export_session(session_id, format_choice, sessions_dir)
            else:
                print("Invalid format. Use 'txt' or 'json'.")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 