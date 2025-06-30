# TravelBot - Your AI Travel Assistant 🌍✈️

TravelBot is an intelligent travel assistant that helps you search for flights, plan trips, and get travel-related information through a natural language interface. Built with Python FastAPI backend and React TypeScript frontend.

## 🌟 Key Features

### 🎯 Intelligent Flight Search
- **Natural Language Processing**: Ask in plain English - "Find flights from Ahmedabad to Kochi for next month"
- **Smart Spell Correction**: Automatically corrects misspelled city names and suggests alternatives
- **Month-Range Search**: Search entire months to find the best deals across multiple dates
- **Multi-Format Support**: Accepts city names, airport codes, and various date formats

### 💫 Enhanced User Experience
- **Interactive Flight Display**: Rich flight information with expandable details
- **Real-Time Progress**: Live progress tracking for month-long searches
- **Smart Suggestions**: Context-aware suggestions based on your queries
- **Month Search Warning**: Confirms long searches with estimated time and benefits

### 🔧 Advanced Backend Features
- **Zero Cache Policy**: Always fresh flight data from live APIs
- **Enhanced Error Handling**: Robust error recovery and detailed debugging
- **Currency Conversion**: Automatic EUR to INR conversion for Indian users
- **Live Airline Data**: Real-time airline information extracted from flight results
- **API Debugging Tools**: Built-in tools for troubleshooting API connectivity

### 🎨 Modern Frontend
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Real-Time Chat Interface**: Instant responses with typing indicators
- **Enhanced Flight Cards**: Detailed flight information with sorting and filtering
- **Progress Animations**: Beautiful loading states for better user experience

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Amadeus API credentials (for flight data)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd travelbot
   ```

2. **Set up Python environment**
   ```bash
   cd api
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the `api` directory:
   ```env
   AMADEUS_API_KEY=your_amadeus_api_key
   AMADEUS_API_SECRET=your_amadeus_api_secret
   OPENAI_API_KEY=your_openai_api_key  # Optional for enhanced spell checking
   ```

4. **Start the backend server**
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

## 📖 Usage Examples

### Basic Flight Search
```
"Find flights from Ahmedabad to Kochi"
"Search AMD to COK for tomorrow"
"Show me flights from Mumbai to Delhi next week"
```

### Month Range Search
```
"Find flights for the entire month of July"
"Show me all flights for next month"
"Search flights in August from BLR to GOI"
```

### Advanced Queries
```
"Business class flights from Chennai to Bangalore"
"Direct flights only from DEL to BOM"
"Cheapest flights from Pune to Hyderabad for 2 passengers"
```

## 🏗️ Architecture

### Backend (`/api`)
- **FastAPI**: High-performance async web framework
- **Amadeus API**: Live flight data integration
- **Enhanced Spell Checker**: City name correction with fuzzy matching
- **Smart Date Parser**: Handles various date formats and month ranges
- **Live Data Manager**: Ensures always-fresh flight information

### Frontend (`/frontend`)
- **React 18 + TypeScript**: Modern React with full type safety
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful, customizable icons

## 🎯 API Endpoints

### Core Endpoints
- `POST /query` - Natural language flight search
- `POST /flights/search` - Direct flight search with parameters
- `GET /airport/{code}` - Airport information
- `GET /airline/{code}` - Airline information

### Utility Endpoints
- `POST /spell-check` - City name spell correction
- `POST /resolve-location` - Location to airport code resolution
- `GET /health` - API health check with data freshness info
- `GET /cities` - Supported cities list

## 🔧 Configuration

### Environment Variables
```env
# Required - Amadeus API
AMADEUS_API_KEY=your_amadeus_api_key
AMADEUS_API_SECRET=your_amadeus_api_secret

# Optional - OpenAI for enhanced spell checking
OPENAI_API_KEY=your_openai_api_key

# Optional - Customization
API_BASE_URL=http://localhost:8000
```

### Frontend Configuration
Update `frontend/src/services/api.ts` to match your backend URL:
```typescript
const API_BASE_URL = 'http://localhost:8000';
```

## 🧪 Testing

### Backend Tests
```bash
cd api
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm run test
```

### Manual Testing
1. Start both backend and frontend servers
2. Try various flight search queries
3. Test month-range searches
4. Verify spell correction with misspelled city names

## 🐛 Troubleshooting

### Common Issues

**422 Error (Unprocessable Entity)**
- Check Amadeus API credentials
- Verify request format in API logs
- Try simplified queries first

**Connection Refused**
- Ensure backend server is running on port 8000
- Check firewall settings
- Verify API_BASE_URL in frontend configuration

**No Flight Results**
- Verify airport codes are correct
- Check if the route exists
- Try different dates or nearby airports

**Month Search Taking Too Long**
- Month searches can take 30-60 seconds
- Check network connectivity
- Monitor backend logs for progress

### Debug Tools
- `GET /debug/cache-status` - Check cache status
- `GET /debug/freshness-test` - Test data freshness validation
- `POST /debug/route` - Debug specific route issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript best practices
- Add proper error handling
- Write tests for new features
- Update documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Amadeus API](https://developers.amadeus.com/) for flight data
- [OpenAI](https://openai.com/) for natural language processing
- [Tailwind CSS](https://tailwindcss.com/) for beautiful styling
- [Lucide](https://lucide.dev/) for amazing icons

## 📞 Support

For support, email support@travelbot.com or join our [Discord community](https://discord.gg/travelbot).

---

**Happy Traveling! 🌍✈️**