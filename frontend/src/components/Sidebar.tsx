import React from 'react';
import { MapPin, Star, Clock, Compass, Settings, HelpCircle, Phone } from 'lucide-react';

const Sidebar: React.FC = () => {
  const menuItems = [
    { icon: Compass, label: 'Dashboard', active: true },
    { icon: MapPin, label: 'My Trips', active: false },
    { icon: Star, label: 'Favorites', active: false },
    { icon: Clock, label: 'History', active: false },
  ];

  const supportItems = [
    { icon: Phone, label: 'Contact Us' },
    { icon: HelpCircle, label: 'Help Center' },
    { icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-sky-500 to-blue-600 rounded-lg flex items-center justify-center">
            <MapPin className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">TravelBot</h1>
            <p className="text-sm text-gray-500">Your AI Travel Assistant</p>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6">
        <div className="space-y-8">
          <div>
            <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">Navigation</h3>
            <nav className="space-y-2">
              {menuItems.map((item, index) => (
                <button
                  key={index}
                  className={`group w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 transform hover:scale-105 hover:shadow-lg ${
                    item.active
                      ? 'bg-sky-50 text-sky-700 border border-sky-200 shadow-sm'
                      : 'text-gray-600 hover:bg-gradient-to-r hover:from-gray-50 hover:to-gray-100 hover:text-gray-900 hover:border hover:border-gray-200'
                  }`}
                >
                  <item.icon className={`w-5 h-5 transition-all duration-300 ${
                    item.active 
                      ? 'text-sky-600' 
                      : 'text-gray-500 group-hover:text-gray-700 group-hover:scale-110'
                  }`} />
                  <span className="transition-all duration-300 group-hover:translate-x-1">{item.label}</span>
                  {!item.active && (
                    <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <div className="w-2 h-2 bg-sky-400 rounded-full"></div>
                    </div>
                  )}
                </button>
              ))}
            </nav>
          </div>

          <div className="bg-gradient-to-br from-sky-50 to-blue-50 rounded-xl p-4 border border-sky-100 hover:shadow-lg transition-all duration-300 transform hover:scale-105">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-orange-400 to-orange-500 rounded-full flex items-center justify-center">
                <Star className="w-4 h-4 text-white" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-gray-900">Premium Support</h4>
                <p className="text-xs text-gray-600">24/7 assistance</p>
              </div>
            </div>
            <p className="text-xs text-gray-600 mb-3">
              Get priority support and exclusive travel deals with our premium service.
            </p>
            <button className="w-full bg-gradient-to-r from-sky-500 to-blue-600 text-white text-xs font-medium py-2 px-3 rounded-lg hover:from-sky-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 hover:shadow-md">
              Upgrade Now
            </button>
          </div>
        </div>
      </div>

      <div className="p-6 border-t border-gray-200">
        <div className="space-y-2">
          {supportItems.map((item, index) => (
            <button
              key={index}
              className="group w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-600 hover:bg-gradient-to-r hover:from-gray-50 hover:to-gray-100 hover:text-gray-900 rounded-lg transition-all duration-300 transform hover:scale-105 hover:shadow-md hover:border hover:border-gray-200"
            >
              <item.icon className="w-4 h-4 transition-all duration-300 group-hover:text-gray-700 group-hover:scale-110" />
              <span className="transition-all duration-300 group-hover:translate-x-1">{item.label}</span>
              <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <div className="w-1.5 h-1.5 bg-gray-400 rounded-full"></div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;