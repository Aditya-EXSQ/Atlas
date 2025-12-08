#!/usr/bin/env python3
"""
Test script to verify the message storage fix.
Simulates a conversation and checks database for balanced storage.
"""

import subprocess
import sqlite3
import time
import os


def send_message(process, message):
    """Send a message to the chatbot."""
    process.stdin.write(message + "\n")
    process.stdin.flush()
    time.sleep(3)  # Wait for response


def main():
    # Test messages - enough to trigger archiving (window = 5 turns)
    messages = [
        "Hi there!",
        "Tell me about Python programming",
        "What are decorators?",
        "Explain async/await",
        "What is a generator?",
        "Tell me about list comprehensions",  # This should trigger archiving
        "What are lambda functions?",
        "Explain closures",
        "exit",
    ]

    print("🚀 Starting chatbot test...")
    print(f"📝 Sending {len(messages) - 1} messages to trigger archiving\n")

    # Start chatbot process
    process = subprocess.Popen(
        ["python", "main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Wait for initialization
    time.sleep(10)

    # Send messages
    for i, msg in enumerate(messages, 1):
        if msg == "exit":
            print(f"\n[{i}/{len(messages)}] Sending exit command...")
            send_message(process, msg)
            break

        print(f"[{i}/{len(messages)}] Sending: {msg}")
        send_message(process, msg)

    # Wait for process to finish
    print("\n⏳ Waiting for chatbot to save and exit...")
    stdout, stderr = process.communicate(timeout=10)

    # Check database
    print("\n" + "=" * 60)
    print("📊 DATABASE VERIFICATION")
    print("=" * 60)

    db_path = "faiss_index/rag_metadata.db"

    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count by type
    print("\n1. Message count by type:")
    cursor.execute("SELECT type, COUNT(*) FROM memories GROUP BY type")
    results = cursor.fetchall()

    user_count = 0
    assistant_count = 0

    for role, count in results:
        print(f"   {role}: {count}")
        if role == "user":
            user_count = count
        elif role == "assistant":
            assistant_count = count

    # Show all messages
    print("\n2. All stored messages:")
    cursor.execute("SELECT id, type, SUBSTR(text, 1, 60) FROM memories ORDER BY id")
    for row in cursor.fetchall():
        print(f"   ID {row[0]:2d} | {row[1]:9s} | {row[2]}")

    # Check for duplicates
    print("\n3. Checking for duplicates:")
    cursor.execute(
        "SELECT text, COUNT(*) as cnt FROM memories GROUP BY text HAVING cnt > 1"
    )
    duplicates = cursor.fetchall()

    if duplicates:
        print("   ❌ Found duplicates:")
        for text, count in duplicates:
            print(f"      '{text[:50]}...' appears {count} times")
    else:
        print("   ✅ No duplicates found")

    conn.close()

    # Verification summary
    print("\n" + "=" * 60)
    print("✅ VERIFICATION SUMMARY")
    print("=" * 60)

    balance_ok = abs(user_count - assistant_count) <= 1
    no_duplicates = len(duplicates) == 0

    print(f"\n✓ Balanced storage: {'PASS' if balance_ok else 'FAIL'}")
    print(f"  User messages: {user_count}, Assistant messages: {assistant_count}")
    print(f"\n✓ No duplicates: {'PASS' if no_duplicates else 'FAIL'}")

    if balance_ok and no_duplicates:
        print("\n🎉 All tests PASSED! The fix is working correctly.")
    else:
        print("\n❌ Some tests FAILED. Please review the results above.")


if __name__ == "__main__":
    main()
