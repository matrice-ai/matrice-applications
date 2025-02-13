
## Use Cases

1. Staff relocating equipment from one location to another
    - **Classification:** Needs Videos for Training
    - **Explanation:** This involves tracking movement and understanding motion patterns, requiring sequential frames. Video datasets are necessary for trajectory analysis.
    - **Video Inferencing:** Required for real-time tracking and action detection.
    - **Industry Deployment:** Not possible with just image inferencing, as movement context is crucial.

2. Detection of oil spills or waste accumulation
    - **Classification:** Possible on Current Platform
    - **Explanation:** Image-based object detection models can identify spills or waste. However, accurate detection depends on varied spill scenarios.
    - **Specific Dataset Requirement:** Niche datasets for oil spills are not easily available, potentially needing custom data collection.
    - **Video Inferencing:** Useful for real-time monitoring and alerts.
    - **Industry Deployment:** Possible with image inferencing, but video enhances detection accuracy by monitoring changes over time.

3. Total time each vehicle spends on the premises versus actual working time
    - **Classification:** Needs Videos for Training
    - **Explanation:** Involves vehicle tracking and temporal analysis to differentiate between idle and active states.
    - **Video Inferencing:** Essential for real-time tracking and time calculations.
    - **Industry Deployment:** Not possible with image inferencing, as temporal context is needed to track duration.

4. Staff not attending to customers (through Staff Uniform Detection)
    - **Classification:** Possible on Current Platform (possible extra logic required)
    - **Explanation:** Uniform detection can be done using image classification and object detection. However, detecting customer interaction requires temporal context, suggesting video training.
    - **Video Inferencing:** Needed for continuous behavior analysis.
    - **Industry Deployment:** Limited functionality with image inferencing (e.g., detecting presence but not interaction). Video is required for comprehensive monitoring.

5. Dress code violations (with face recognition)
    - **Classification:** Needs Videos for Training, can be done with images only but limited accuracy.
    - **Explanation:** Involves consistent tracking of attire and face recognition across frames.
    - **Specific Dataset Requirement:** Requires datasets with varied dress codes and environmental conditions.
    - **Video Inferencing:** Required for continuous monitoring and alerts.
    - **Industry Deployment:** Not possible with image inferencing alone, as tracking consistency is required.

6. Vehicle wait time for service (SLA monitoring)
    - **Classification:** Needs Videos for Training
    - **Explanation:** Requires tracking vehicle entry and exit to measure wait times accurately.
    - **Video Inferencing:** Necessary for real-time SLA monitoring and alerts.
    - **Industry Deployment:** Not possible with image inferencing, as temporal context is crucial for duration calculation.

7. Smoke detection (Face Detection)
    - **Classification:** Possible on Current Platform
    - **Explanation:** Smoke can be detected using image segmentation and object detection models. Face detection is not necessary unless identifying the smoker.
    - **Specific Dataset Requirement:** Datasets for smoke detection are niche and may require custom data collection.
    - **Video Inferencing:** Useful for real-time alerts.
    - **Industry Deployment:** Possible with image inferencing but enhanced accuracy with video for monitoring the spread and movement of smoke.

8. Animal detection in the service area (e.g., cat, dog)
    - **Classification:** Possible on Current Platform (Extra logic maybe req - geofencing)
    - **Explanation:** Object detection models can identify animals using images.
    - **Specific Dataset Requirement:** Requires datasets covering different animal appearances and poses.
    - **Video Inferencing:** Useful for real-time alerts and movement tracking.
    - **Industry Deployment:** Possible with image inferencing for static detection but video is beneficial for tracking movement.

9. Staff using mobile phones in the presence of customers
    - **Classification:** Needs Videos for Training
    - **Explanation:** This involves action recognition and interaction detection, requiring temporal context.
    - **Video Inferencing:** Required for continuous behavioral analysis.
    - **Industry Deployment:** Not possible with image inferencing as temporal context is needed for action recognition.

10. Improper storage of stock (cartons) in the service area
     - **Classification:** Possible on Current Platform
     - **Explanation:** Object detection models can identify improper storage arrangements.
     - **Video Inferencing:** Useful for continuous monitoring and alerts.
     - **Industry Deployment:** Possible with image inferencing but enhanced with video for detecting gradual stock movement or buildup.

11. Detection of activities outside of working hours
     - **Classification:** Needs Videos for Training
     - **Explanation:** Temporal analysis is required to differentiate between normal and abnormal activities.
     - **Video Inferencing:** Necessary for real-time security monitoring.
     - **Industry Deployment:** Not possible with image inferencing, as time-sequenced activity recognition is required.

12. Face detection and recognition
     - **Classification:** Possible on Current Platform
     - **Explanation:** Image-based face detection and recognition models are readily available.
     - **Specific Dataset Requirement:** Requires diverse datasets with varied lighting, angles, and demographics.
     - **Video Inferencing:** Useful for continuous identity verification.
     - **Industry Deployment:** Possible with image inferencing, but video improves accuracy by reducing false positives/negatives through tracking.

13. People counting
     - **Classification:** Possible on Current Platform, but Video Enhances Accuracy (Extra logic is required on top of current models)
     - **Explanation:** Image-based models can estimate count, but video tracking prevents double counting by tracking movement.
     - **Video Inferencing:** Enhances accuracy and real-time monitoring.
     - **Industry Deployment:** Possible with image inferencing, but video significantly improves accuracy and flow tracking.

14. Vehicle detection
     - **Classification:** Possible on Current Platform
     - **Explanation:** Object detection models can be used for vehicle identification.
     - **Video Inferencing:** Useful for real-time traffic monitoring and alerts.
     - **Industry Deployment:** Possible with image inferencing, but video aids in tracking movement and flow.

15. License plate recognition
     - **Classification:** Possible on Current Platform
     - **Explanation:** Image-based OCR models can recognize license plates. (working on OCR)
     - **Specific Dataset Requirement:** Requires varied datasets with different angles, lighting, and fonts.
     - **Video Inferencing:** Useful for tracking vehicles over time.
     - **Industry Deployment:** Possible with image inferencing but enhanced with video for tracking and sequence validation.