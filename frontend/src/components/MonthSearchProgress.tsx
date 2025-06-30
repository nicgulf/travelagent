import React, { useEffect, useState } from 'react';
import { Calendar, Clock, Search, Plane, TrendingUp } from 'lucide-react';

interface MonthSearchProgressProps {
  currentDate: number;
  totalDates: number;
  monthName: string;
  currentSearchDate?: string;
  flightsFound: number;
  isVisible: boolean;
}

export const MonthSearchProgress: React.FC<MonthSearchProgressProps> = ({
  currentDate,
  totalDates,
  monthName,
  currentSearchDate,
  flightsFound,
  isVisible
}) => {
  const [animatedProgress, setAnimatedProgress] = useState(0);
  const progressPercentage = (currentDate / totalDates) * 100;
  
  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(() => {
        setAnimatedProgress(progressPercentage);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [progressPercentage, isVisible]);

  if (!isVisible) return null;
  
  return (
    <div className="fixed inset-0 bg-white bg-opacity-98 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="text-center max-w-lg mx-4">
        {/* Animated Icon */}
        <div className="relative mb-8">
          <div className="animate-bounce">
            <Plane className="mx-auto text-blue-600" size={52} />
          </div>
          <div className="absolute -top-2 -right-2 animate-pulse">
            <Search className="text-blue-400 bg-white rounded-full p-1" size={24} />
          </div>
        </div>
        
        {/* Title */}
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          Searching {monthName}
        </h2>
        
        <p className="text-gray-600 mb-8 text-lg">
          Finding the best flight deals across the month...
        </p>
        
        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-4 mb-6 overflow-hidden">
          <div 
            className="bg-gradient-to-r from-blue-500 via-blue-600 to-green-500 h-4 rounded-full transition-all duration-1000 ease-out relative"
            style={{ width: `${animatedProgress}%` }}
          >
            {/* Animated shine effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
          </div>
        </div>
        
        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-6 mb-8">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-3xl font-bold text-blue-600">{currentDate}</div>
            <div className="text-sm text-gray-600 mt-1">Dates Searched</div>
          </div>
          
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-3xl font-bold text-green-600">{flightsFound}</div>
            <div className="text-sm text-gray-600 mt-1">Flights Found</div>
          </div>
          
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl font-bold text-gray-600">{totalDates - currentDate}</div>
            <div className="text-sm text-gray-600 mt-1">Remaining</div>
          </div>
        </div>
        
        {/* Current Status */}
        {currentSearchDate && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 mb-6 border border-blue-200">
            <div className="flex items-center justify-center text-blue-700">
              <Clock className="mr-2 animate-spin" size={16} />
              <span className="text-sm">
                Currently searching: <strong>{new Date(currentSearchDate).toLocaleDateString('en-US', { 
                  weekday: 'short', 
                  month: 'short', 
                  day: 'numeric' 
                })}</strong>
              </span>
            </div>
          </div>
        )}
        
        {/* Dynamic Encouraging Messages */}
        <div className="mb-4">
          {currentDate === 0 && (
            <p className="text-gray-600 animate-fade-in">🚀 Starting comprehensive search...</p>
          )}
          {currentDate > 0 && currentDate < totalDates / 3 && (
            <p className="text-gray-600 animate-fade-in">🔍 Great deals are being discovered...</p>
          )}
          {currentDate >= totalDates / 3 && currentDate < (totalDates * 2) / 3 && (
            <p className="text-gray-600 animate-fade-in">✈️ Finding amazing flight options...</p>
          )}
          {currentDate >= (totalDates * 2) / 3 && currentDate < totalDates && (
            <p className="text-gray-600 animate-fade-in">🎯 Almost done! Finalizing your results...</p>
          )}
          {currentDate === totalDates && (
            <p className="text-green-600 font-semibold animate-fade-in">✅ Search complete! Preparing results...</p>
          )}
        </div>
        
        {/* Progress Text */}
        <div className="text-xs text-gray-400">
          {Math.round(animatedProgress)}% complete • Estimated time remaining: {Math.max(0, Math.round((totalDates - currentDate) * 2.5))} seconds
        </div>
        
        {/* Fun fact */}
        <div className="mt-6 p-3 bg-yellow-50 rounded-lg border-l-4 border-yellow-400">
          <p className="text-sm text-yellow-700">
            💡 Did you know? We're checking multiple airlines and routes to ensure you get the best price!
          </p>
        </div>
      </div>
    </div>
  );
};