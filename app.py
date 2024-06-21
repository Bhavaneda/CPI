const mongoose = require('mongoose');

const WatchlistSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    stocks: [
        {
            ticker: {
                type: String,
                required: true,
            },
            notes: String,
        },
    ],
});

module.exports = mongoose.model('Watchlist', WatchlistSchema);





const express = require('express');
const router = express.Router();
const Watchlist = require('../models/Watchlist');

// Create a new watchlist
router.post('/create', async (req, res) => {
    const { name } = req.body;
    try {
        const newWatchlist = new Watchlist({ name, stocks: [] });
        const watchlist = await newWatchlist.save();
        res.json(watchlist);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Add a stock to a watchlist
router.post('/add', async (req, res) => {
    const { watchlistId, ticker } = req.body;
    try {
        const watchlist = await Watchlist.findById(watchlistId);
        if (!watchlist) {
            return res.status(404).json({ msg: 'Watchlist not found' });
        }
        watchlist.stocks.push({ ticker });
        await watchlist.save();
        res.json(watchlist);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Remove a stock from a watchlist
router.post('/remove', async (req, res) => {
    const { watchlistId, ticker } = req.body;
    try {
        const watchlist = await Watchlist.findById(watchlistId);
        if (!watchlist) {
            return res.status(404).json({ msg: 'Watchlist not found' });
        }
        watchlist.stocks = watchlist.stocks.filter(stock => stock.ticker !== ticker);
        await watchlist.save();
        res.json(watchlist);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Get all watchlists
router.get('/', async (req, res) => {
    try {
        const watchlists = await Watchlist.find();
        res.json(watchlists);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

module.exports = router;







const express = require('express');
const mongoose = require('mongoose');
const watchlistRoutes = require('./routes/watchlist');
const dotenv = require('dotenv');

dotenv.config();

const app = express();

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
}).then(() => console.log('MongoDB connected...'))
  .catch(err => console.error(err.message));

// Middleware
app.use(express.json());

// Routes
app.use('/api/watchlist', watchlistRoutes);

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => console.log(`Server started on port ${PORT}`));















import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Watchlist = () => {
    const [watchlists, setWatchlists] = useState([]);
    const [newWatchlistName, setNewWatchlistName] = useState('');
    const [selectedWatchlistId, setSelectedWatchlistId] = useState(null);
    const [newTicker, setNewTicker] = useState('');

    useEffect(() => {
        fetchWatchlists();
    }, []);

    const fetchWatchlists = async () => {
        try {
            const res = await axios.get('/api/watchlist');
            setWatchlists(res.data);
        } catch (err) {
            console.error(err.message);
        }
    };

    const createWatchlist = async () => {
        try {
            const res = await axios.post('/api/watchlist/create', { name: newWatchlistName });
            setWatchlists([...watchlists, res.data]);
            setNewWatchlistName('');
        } catch (err) {
            console.error(err.message);
        }
    };

    const addStockToWatchlist = async (watchlistId) => {
        try {
            const res = await axios.post('/api/watchlist/add', { watchlistId, ticker: newTicker });
            setWatchlists(watchlists.map(w => w._id === watchlistId ? res.data : w));
            setNewTicker('');
        } catch (err) {
            console.error(err.message);
        }
    };

    const removeStockFromWatchlist = async (watchlistId, ticker) => {
        try {
            const res = await axios.post('/api/watchlist/remove', { watchlistId, ticker });
            setWatchlists(watchlists.map(w => w._id === watchlistId ? res.data : w));
        } catch (err) {
            console.error(err.message);
        }
    };

    return (
        <div>
            <h1>Watchlists</h1>
            <div>
                <input
                    type="text"
                    value={newWatchlistName}
                    onChange={(e) => setNewWatchlistName(e.target.value)}
                    placeholder="New Watchlist Name"
                />
                <button onClick={createWatchlist}>Create Watchlist</button>
            </div>
            {watchlists.map((watchlist) => (
                <div key={watchlist._id}>
                    <h2>{watchlist.name}</h2>
                    <ul>
                        {watchlist.stocks.map((stock) => (
                            <li key={stock.ticker}>
                                {stock.ticker}
                                <button onClick={() => removeStockFromWatchlist(watchlist._id, stock.ticker)}>Remove</button>
                            </li>
                        ))}
                    </ul>
                    <input
                        type="text"
                        value={newTicker}
                        onChange={(e) => setNewTicker(e.target.value)}
                        placeholder="Add Stock Ticker"
                    />
                    <button onClick={() => addStockToWatchlist(watchlist._id)}>Add Stock</button>
                </div>
            ))}
        </div>
    );
};

export default Watchlist;











import React from 'react';
import Watchlist from './components/Watchlist';

function App() {
    return (
        <div className="App">
            <Watchlist />
        </div>
    );
}

export default App;



import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';

ReactDOM.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
    document.getElementById('root')
);




