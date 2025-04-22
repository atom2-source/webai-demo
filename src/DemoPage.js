import React, { useState, useRef, useEffect } from 'react';
import './MainContent.css';
import './DemoPage.css';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

function DemoPage({ setActivePage }) {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [model, setModel] = useState(null);
  const [videoStream, setVideoStream] = useState(null);
  const [detections, setDetections] = useState([]);
  const [enhancedDetections, setEnhancedDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [detectionMode, setDetectionMode] = useState('camera'); // 'camera' or 'image'
  const [detectionRunning, setDetectionRunning] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const animationRef = useRef(null);
  
  // Load the COCO-SSD model on component mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        
        // Explicitly set the backend to WebGL (faster) with CPU fallback
        try {
          await import('@tensorflow/tfjs-backend-webgl');
          await import('@tensorflow/tfjs-backend-cpu');
          const tf = await import('@tensorflow/tfjs');
          await tf.setBackend('webgl');
          console.log("Using WebGL backend");
        } catch (backendError) {
          console.warn("Failed to set WebGL backend, falling back to CPU", backendError);
          const tf = await import('@tensorflow/tfjs');
          await tf.setBackend('cpu');
          console.log("Using CPU backend");
        }
        
        // Load the model
        const loadedModel = await cocoSsd.load();
        console.log("COCO-SSD Model loaded successfully");
        setModel(loadedModel);
        setIsModelLoading(false);
      } catch (error) {
        console.error("Error loading COCO-SSD model:", error);
        setIsModelLoading(false);
      }
    };
    
    loadModel();
    
    // Cleanup function to handle component unmount
    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Start webcam when model is loaded and camera mode is selected
  useEffect(() => {
    if (!isModelLoading && model && detectionMode === 'camera') {
      startWebcam();
    }
    
    // Cleanup function for video stream when switching modes
    return () => {
      if (videoStream && detectionMode !== 'camera') {
        videoStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isModelLoading, model, detectionMode, videoStream]);
  
  // Start the webcam
  const startWebcam = async () => {
    try {
      const constraints = {
        video: {
          width: 640,
          height: 480,
          facingMode: 'environment'
        }
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setVideoStream(stream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Error starting webcam:", error);
    }
  };
  
  // Draw bounding boxes and labels on canvas
  const drawDetections = (predictions, isImageMode = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw video or image on canvas
    if (isImageMode && imageRef.current) {
      canvas.width = imageRef.current.width;
      canvas.height = imageRef.current.height;
      ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    } else if (videoRef.current) {
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    }
    
    // Draw each prediction
    predictions.forEach(prediction => {
      // Get prediction details
      const [x, y, width, height] = prediction.bbox;
      const text = `${prediction.class} ${Math.round(prediction.score * 100)}%`;
      
      // Draw bounding box
      ctx.strokeStyle = '#00FFFF';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      
      // Draw background for text
      ctx.fillStyle = '#00FFFF';
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x, y - 25, textWidth + 10, 25);
      
      // Draw text
      ctx.fillStyle = '#000000';
      ctx.font = '18px Arial';
      ctx.fillText(text, x + 5, y - 7);
    });
  };
  
  // Handle video play
  const handleVideoPlay = () => {
    if (canvasRef.current && videoRef.current && model) {
      // Make sure video is actually playing and has dimensions before starting detection
      if (videoRef.current.readyState >= 2) {
        setDetectionRunning(true);
        runDetection();
      } else {
        // Wait for video to be ready before starting detection
        console.log("Waiting for video to be ready...");
        const checkVideoReady = () => {
          if (videoRef.current && videoRef.current.readyState >= 2) {
            setDetectionRunning(true);
            runDetection();
          } else {
            setTimeout(checkVideoReady, 100); // Check again in 100ms
          }
        };
        checkVideoReady();
      }
    }
  };
  
  // Run detection on video frames
  const runDetection = async () => {
    if (!model || !videoRef.current || !canvasRef.current || !detectionRunning) {
      return;
    }
    
    // Check if video has valid dimensions and is actually playing
    if (
      videoRef.current.readyState !== 4 || 
      videoRef.current.videoWidth === 0 || 
      videoRef.current.videoHeight === 0
    ) {
      // Video not ready yet, try again in the next frame
      animationRef.current = requestAnimationFrame(runDetection);
      return;
    }
    
    try {
      // Run detection
      const predictions = await model.detect(videoRef.current, 0.4);
      if (predictions && predictions.length > 0) {
        console.log("Detected objects:", predictions);
        setDetections(predictions);
        
        // Draw detections
        drawDetections(predictions);
      }
      
      // Continue detection loop
      if (detectionRunning) {
        animationRef.current = requestAnimationFrame(runDetection);
      }
    } catch (error) {
      console.error("Error in object detection:", error);
      // Try again after a short delay if still running
      if (detectionRunning) {
        setTimeout(() => {
          animationRef.current = requestAnimationFrame(runDetection);
        }, 1000); // Wait 1 second before trying again
      }
    }
  };
  
  // Toggle detection on/off for camera mode
  const toggleDetection = () => {
    if (detectionRunning) {
      // Stop detection
      setDetectionRunning(false);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    } else if (detectionMode === 'camera' && videoRef.current && model) {
      // Start detection
      setDetectionRunning(true);
      // Start detection in the next event loop tick
      setTimeout(runDetection, 0);
    }
  };
  
  // Handle image upload
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target.result);
      setDetectionMode('image');
    };
    reader.readAsDataURL(file);
  };
  
  // Detect objects in the uploaded image
  const detectObjectsInImage = async () => {
    if (!model || !imageRef.current) return;
    
    // Check if image has valid dimensions and is fully loaded
    if (
      !imageRef.current.complete || 
      imageRef.current.naturalWidth === 0 ||
      imageRef.current.naturalHeight === 0
    ) {
      console.log("Image not fully loaded yet, waiting...");
      setTimeout(detectObjectsInImage, 100); // Try again after a short delay
      return;
    }
    
    try {
      // Run detection on the image
      const predictions = await model.detect(imageRef.current, 0.4);
      console.log("Image detection results:", predictions);
      setDetections(predictions);
      
      // Draw detections on the canvas
      drawDetections(predictions, true);
    } catch (error) {
      console.error("Error detecting objects in image:", error);
    }
  };
  
  // Effect to run detection when image is loaded
  useEffect(() => {
    if (selectedImage && imageRef.current && model && detectionMode === 'image') {
      // Wait for the image to fully load before detecting
      const img = imageRef.current;
      if (img.complete) {
        detectObjectsInImage();
      } else {
        img.onload = detectObjectsInImage;
      }
    }
  }, [selectedImage, model, detectionMode]);
  
  // Process detections with backend server - now triggered by button click
  const processDetectionsWithBackend = async () => {
    if (detections.length === 0) {
      alert("No objects detected to process.");
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Prepare data for server
      const detectionData = {
        detections: detections,
        imageContext: {
          width: canvasRef.current.width,
          height: canvasRef.current.height,
          timestamp: new Date().toISOString()
        }
      };
      
      // Make API call to your backend server
      const response = await fetch('https://webai-server-44469913499.us-central1.run.app/api/process-detection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(detectionData)
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Update enhanced detections
      if (data.enhancedDetections) {
        setEnhancedDetections(data.enhancedDetections);
      }
    } catch (error) {
      console.error("Error processing detections with backend:", error);
      
      // Generate simulated responses for demo purposes when server is not available
      const simulatedEnhancedDetections = detections.map(detection => {
        return {
          ...detection,
          enhanced: {
            detailedDescription: `${detection.class.charAt(0).toUpperCase() + detection.class.slice(1)} typically found in everyday environments.`,
            objectInfo: {
              purpose: `Common object used by people for specific functions`,
              capabilities: `Performs tasks related to its design and purpose`,
              features: `Has distinctive characteristics typical of this category`
            }
          }
        };
      });
      
      setEnhancedDetections(simulatedEnhancedDetections);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Switch between camera and image upload modes
  const switchMode = (mode) => {
    if (mode === detectionMode) return;
    
    // Clean up previous mode
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    
    setDetectionMode(mode);
    setDetections([]);
    setEnhancedDetections([]);
    setDetectionRunning(false);
    
    if (mode === 'camera' && !videoStream) {
      startWebcam();
    } else if (mode === 'image') {
      // Stop camera if it's running
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        setVideoStream(null);
      }
    }
  };
  
  return (
    <div className="container">
      <div className="sidebar">
        <div 
          className="sidebar-item" 
          onClick={() => setActivePage('intro')}
        >
          <div className="icon">
            <span>i</span>
          </div>
          <div className="label">Intro</div>
        </div>
        <div 
          className="sidebar-item" 
          onClick={() => setActivePage('techstack')}
        >
          <div className="icon">
            <span>&lt;/&gt;</span>
          </div>
          <div className="label">Tech Stack</div>
        </div>
        <div 
          className="sidebar-item active" 
          onClick={() => setActivePage('demo')}
        >
          <div className="icon">
            <span>▶</span>
          </div>
          <div className="label">Demo</div>
        </div>
      </div>
      <div className="main-content">
        <div className="demo-container">
          <h1>Object Detection Demo</h1>
          
          {isModelLoading ? (
            <div className="loading-container">
              <p>Loading TensorFlow.js model...</p>
              <div className="loading-spinner"></div>
            </div>
          ) : (
            <>
              <div className="mode-switcher">
                <button 
                  className={`mode-button ${detectionMode === 'camera' ? 'active' : ''}`}
                  onClick={() => switchMode('camera')}
                >
                  Camera
                </button>
                <button 
                  className={`mode-button ${detectionMode === 'image' ? 'active' : ''}`}
                  onClick={() => switchMode('image')}
                >
                  Image Upload
                </button>
              </div>
              
              <div className="detection-container">
                <div className="video-container">
                  {detectionMode === 'camera' ? (
                    <>
                      <video 
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        onPlay={handleVideoPlay}
                      />
                      <div className="control-overlay">
                        <button 
                          className={`control-button ${detectionRunning ? 'active' : ''}`}
                          onClick={toggleDetection}
                        >
                          {detectionRunning ? 'Pause Detection' : 'Start Detection'}
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <input 
                        type="file" 
                        accept="image/*" 
                        onChange={handleImageUpload} 
                        className="image-upload-input"
                      />
                      {selectedImage && (
                        <img 
                          ref={imageRef}
                          src={selectedImage} 
                          alt="Uploaded" 
                          className="uploaded-image" 
                        />
                      )}
                    </>
                  )}
                  <canvas ref={canvasRef} className="detection-canvas" />
                </div>
                
                <div className="results-container">
                  <h2>Detection Results</h2>
                  
                  {detections.length > 0 && (
                    <div className="server-processing-controls">
                      <button 
                        className="cloud-process-button"
                        onClick={processDetectionsWithBackend}
                        disabled={isProcessing}
                      >
                        {isProcessing ? 'Processing...' : 'Send to Cloud AI'}
                      </button>
                      <p className="detection-count">
                        {detections.length} object{detections.length !== 1 ? 's' : ''} detected
                      </p>
                    </div>
                  )}
                  
                  {isProcessing && (
                    <div className="processing-indicator">
                      <div className="loading-spinner small"></div>
                      <span>Processing with cloud model...</span>
                    </div>
                  )}
                  
                  {enhancedDetections.length > 0 ? (
                    <div className="detection-cards">
                      {enhancedDetections.map((detection, index) => (
                        <div key={index} className="detection-card">
                          <h3>{detection.class}</h3>
                          <div className="card-content">
                            <div className="detection-info">
                              <p><strong>Confidence:</strong> {Math.round(detection.score * 100)}%</p>
                              {detection.enhanced && (
                                <>
                                  <p><strong>Description:</strong> {detection.enhanced.detailedDescription}</p>
                                  {detection.enhanced.objectInfo && (
                                    <div className="object-info">
                                      <h4>Object Information:</h4>
                                      <ul>
                                        {Object.entries(detection.enhanced.objectInfo).map(([key, value], i) => (
                                          <li key={i}>
                                            <strong>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:</strong> {value}
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="no-detections">
                      {detections.length === 0 ? (
                        <>
                          <p>No objects detected. Try pointing the camera at an object or uploading a different image.</p>
                          <p className="tip">Tip: Make sure objects are clearly visible with good lighting. The model works best with common objects like people, cars, furniture, and animals.</p>
                        </>
                      ) : (
                        <p>Click "Send to Cloud AI" to get enhanced information about detected objects.</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
          
          <div className="demo-info">
            <h2>How It Works</h2>
            <p>This demo uses TensorFlow.js to run object detection directly in your browser. When you click "Send to Cloud AI", the detection information is sent to our backend server which:</p>
            <ol>
              <li>Securely stores and manages the API key</li>
              <li>Connects to Google's Gemini AI model for enhanced analysis</li>
              <li>Returns detailed information about the detected objects</li>
            </ol>
            <p>The enhanced data includes detailed descriptions, attribute estimations, and practical information about the objects that demonstrates how this system could be used in a production environment.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DemoPage;