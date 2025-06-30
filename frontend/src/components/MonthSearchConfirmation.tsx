import React from 'react';
import { Clock, Calendar, TrendingDown, X } from 'lucide-react';

interface MonthSearchConfirmationProps {
  onConfirm: () => void;
  onCancel: () => void;
  monthName: string;
  estimatedDates: number;
}

export const MonthSearchConfirmation: React.FC<MonthSearchConfirmationProps> = ({
  onConfirm,
  onCancel,
  monthName,
  estimatedDates
}) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 max-w-md mx-4 shadow-2xl relative">
        {/* Close button */}
        <button 
          onClick={onCancel}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
        >
          <X size={20} />
        </button>
        
        <div className="flex items-center mb-4">
          <Calendar className="text-blue-500 mr-3" size={28} />
          <h3 className="text-xl font-bold text-gray-800">
            Searching Entire {monthName}
          </h3>
        </div>
        
        <div className="space-y-3 mb-6">
          <div className="flex items-center text-gray-600">
            <Clock className="mr-2 text-orange-500" size={18} />
            <span>This will take more than 30-60 seconds</span>
          </div>
          
          <div className="flex items-center text-gray-600">
            <TrendingDown className="mr-2 text-green-500" size={18} />
            <span>We'll check {estimatedDates} strategic dates</span>
          </div>
        </div>
        
        <p className="text-gray-600 mb-6">
          Month searches take longer time than we expect sometimes but find the <strong className="text-blue-600">best deals</strong> across 
          all dates in {monthName}.
        </p>
        
        <div className="space-y-3">
          <button
            onClick={onConfirm}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-3 px-4 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 font-medium shadow-lg"
          >
            🚀 Yes, find the best deals
          </button>
          
          <button
            onClick={onCancel}
            className="w-full bg-gray-100 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-200 transition-colors border"
          >
            📅 Let me choose a specific date
          </button>
        </div>
        
        <div className="mt-4 p-3 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg border-l-4 border-blue-400">
          <p className="text-sm text-blue-700">
            💡 <strong>Pro tip:</strong> Month searches often save 20-40% compared to single dates
          </p>
        </div>
      </div>
    </div>
  );
};