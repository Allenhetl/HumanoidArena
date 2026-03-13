"""
Script to simplify pico_server_sonic.py - only keep POSE mode
"""

import re

def simplify_pico_server():
    with open('pico_server_sonic.py', 'r') as f:
        content = f.read()
    
    # Lines to keep (approximate ranges based on analysis)
    # Keep imports (1-100)
    # Keep OFFSETS and helper functions (125-450)
    # Keep PicoReader (766-832)
    # Keep ThreePointPose (893-1187)
    # Keep PoseStreamer (1189-1517)
    # Keep run_pico (1519-1574)
    # Simplify main (2079-2218)
    
    lines = content.split('\n')
    
    # Classes/functions to remove
    remove_patterns = [
        r'class LocomotionMode',
        r'class YawAccumulator',
        r'class FeedbackReader',
        r'class PlannerStreamer',
        r'def run_pico_manager',
        r'def run_vr3pt_visualizer_test',
        r'def run_vr3pt_live_visualizer',
        r'def run_vr3pt_realtime_visualizer',
    ]
    
    print("Simplification patterns identified")
    print("Please manually edit the file to:")
    print("1. Remove LocomotionMode enum (lines ~100-123)")
    print("2. Simplify StreamMode to only POSE")
    print("3. Remove YawAccumulator class (lines ~543-582)")
    print("4. Remove FeedbackReader class (lines ~1576-1639)")
    print("5. Remove PlannerStreamer class (lines ~1641-1825)")
    print("6. Remove run_pico_manager function (lines ~1827-2077)")
    print("7. Simplify main to only call run_pico")
    print("8. Remove --manager and feedback-related args")

if __name__ == '__main__':
    simplify_pico_server()
