import React from 'react';
import { Plane, Hotel, MapPin, Calendar, CreditCard, HelpCircle } from 'lucide-react';
import { QuickAction } from '../types/chat';

interface QuickActionsProps {
  onActionClick: (action: string) => void;
}

const QuickActions: React.FC<QuickActionsProps> = ({ onActionClick }) => {
  const actions: QuickAction[] = [
    { id: '1', label: 'Book Flight', icon: 'plane', action: 'I need help booking a flight' },
    { id: '2', label: 'Find Hotels', icon: 'hotel', action: 'Show me available hotels' },
    { id: '3', label: 'Plan Trip', icon: 'map', action: 'Help me plan a trip' },
    { id: '4', label: 'Check Dates', icon: 'calendar', action: 'Check travel dates availability' },
    { id: '5', label: 'Payment Help', icon: 'card', action: 'I need help with payment' },
    { id: '6', label: 'General Help', icon: 'help', action: 'I need general assistance' },
  ];

  const getIcon = (iconName: string) => {
    const icons = {
      plane: Plane,
      hotel: Hotel,
      map: MapPin,
      calendar: Calendar,
      card: CreditCard,
      help: HelpCircle,
    };
    const IconComponent = icons[iconName as keyof typeof icons];
    return <IconComponent className="w-4 h-4" />;
  };

  return (
    <div className="bg-gray-50 border-t border-gray-200 p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Actions</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
        {actions.map((action) => (
          <button
            key={action.id}
            onClick={() => onActionClick(action.action)}
            className="group flex items-center space-x-2 p-3 bg-white rounded-lg border border-green-200 hover:border-green-400 hover:bg-gradient-to-r hover:from-green-50 hover:to-emerald-50 transition-all duration-300 text-sm font-medium text-green-700 hover:text-green-800 shadow-sm hover:shadow-lg transform hover:scale-105 hover:-translate-y-1"
          >
            <div className="text-green-600 transition-all duration-300 group-hover:text-green-700 group-hover:scale-110 group-hover:rotate-12">
              {getIcon(action.icon)}
            </div>
            <span className="transition-all duration-300 group-hover:font-semibold">{action.label}</span>
            <div className="ml-auto opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-x-2 group-hover:translate-x-0">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickActions;