/* eslint-disable no-unused-vars */
import React, { useState, useRef, useEffect } from 'react';

const MyVision = () => {
  const [description, setDescription] = useState('Initializing camera...');
  const videoRef = useRef(null);

  useEffect(() => {
    let stream = null;

    // Immediately try to start camera on component
    const enableCamera = async () => {
      try {
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Once video is ready, update status
          videoRef.current.onloadeddata = () => {
            setDescription('System ready for detection...');
          };
        }
      } catch (err) {
        // In case of any errors, retry after a short delay
        setTimeout(enableCamera, 1000);
      }
    };

    // Start camera access immediately
    enableCamera();

    // Cleanup
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-600 p-4 flex items-center justify-center">
      <div className="w-full max-w-4xl bg-slate-600 rounded-lg overflow-hidden">
        <div className="flex items-center px-4 py-2">
          <div className="w-6 h-6">
            <svg viewBox="0 0 24 24" className="w-full h-full text-white fill-current">
              <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z" />
            </svg>
          </div>
          <h1 className="text-white text-xl ml-2">BeMyVision</h1>
        </div>

        {/* Main Camera Feed Area */}
        <div className="aspect-video bg-gray-200 rounded-lg mx-4 mb-4 relative overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute top-0 left-0 w-full h-full object-cover"
          />
        </div>

        {/* Description Text Area */}
        <div className="mx-4 mb-4">
          <div className="bg-gray-700 text-gray-200 rounded-lg p-4">
            <p className="text-center italic">
              {description}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MyVision;