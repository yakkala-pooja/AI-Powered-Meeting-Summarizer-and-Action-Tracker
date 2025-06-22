import React, { useState, useMemo, createContext, useContext } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import TranscriptAnalyzer from './components/TranscriptAnalyzer';
import ResultsPage from './components/ResultsPage';

// Create a context for theme mode
export const ColorModeContext = createContext({ 
  toggleColorMode: () => {},
  mode: 'light'
});

function App() {
  const [mode, setMode] = useState('light');
  
  // Color mode context value
  const colorMode = useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
      },
      mode,
    }),
    [mode],
  );

  // Create theme based on mode
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: '#2F80ED',
            light: '#5B9AEF',
            dark: '#1A56B3',
          },
          secondary: {
            main: '#F2994A',
            light: '#F4B27A',
            dark: '#D97B29',
          },
          background: {
            default: mode === 'light' ? '#F7FAFC' : '#121212',
            paper: mode === 'light' ? '#FFFFFF' : '#1E1E1E',
          },
          text: {
            primary: mode === 'light' ? '#37352F' : '#E0E0E0',
            secondary: mode === 'light' ? '#6B7280' : '#A0A0A0',
          },
          error: {
            main: '#E53E3E',
          },
          warning: {
            main: '#F6AD55',
          },
          info: {
            main: '#63B3ED',
          },
          success: {
            main: '#48BB78',
          },
          divider: mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)',
        },
        typography: {
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
          h1: {
            fontWeight: 700,
          },
          h2: {
            fontWeight: 700,
          },
          h3: {
            fontWeight: 600,
          },
          h4: {
            fontWeight: 600,
          },
          h5: {
            fontWeight: 600,
          },
          h6: {
            fontWeight: 500,
          },
          subtitle1: {
            fontWeight: 400,
          },
          body1: {
            fontWeight: 400,
            lineHeight: 1.5,
          },
        },
        shape: {
          borderRadius: 8,
        },
        shadows: [
          'none',
          '0px 1px 2px rgba(0, 0, 0, 0.06), 0px 1px 3px rgba(0, 0, 0, 0.1)',
          '0px 1px 5px rgba(0, 0, 0, 0.05), 0px 1px 8px rgba(0, 0, 0, 0.1)',
          '0px 2px 4px rgba(0, 0, 0, 0.05), 0px 3px 6px rgba(0, 0, 0, 0.08)',
          '0px 4px 8px rgba(0, 0, 0, 0.04), 0px 6px 12px rgba(0, 0, 0, 0.08)',
          '0px 5px 15px rgba(0, 0, 0, 0.08)',
          '0px 6px 18px rgba(0, 0, 0, 0.1)',
          '0px 7px 20px rgba(0, 0, 0, 0.12)',
          '0px 8px 22px rgba(0, 0, 0, 0.14)',
          '0px 9px 25px rgba(0, 0, 0, 0.16)',
          '0px 10px 28px rgba(0, 0, 0, 0.18)',
          '0px 11px 30px rgba(0, 0, 0, 0.2)',
          '0px 12px 32px rgba(0, 0, 0, 0.22)',
          '0px 13px 34px rgba(0, 0, 0, 0.24)',
          '0px 14px 36px rgba(0, 0, 0, 0.26)',
          '0px 15px 38px rgba(0, 0, 0, 0.28)',
          '0px 16px 40px rgba(0, 0, 0, 0.3)',
          '0px 17px 42px rgba(0, 0, 0, 0.32)',
          '0px 18px 44px rgba(0, 0, 0, 0.34)',
          '0px 19px 46px rgba(0, 0, 0, 0.36)',
          '0px 20px 48px rgba(0, 0, 0, 0.38)',
          '0px 21px 50px rgba(0, 0, 0, 0.4)',
          '0px 22px 52px rgba(0, 0, 0, 0.42)',
          '0px 23px 54px rgba(0, 0, 0, 0.44)',
          '0px 24px 56px rgba(0, 0, 0, 0.46)',
        ],
        components: {
          MuiButton: {
            styleOverrides: {
              root: {
                textTransform: 'none',
                fontWeight: 500,
                padding: '8px 16px',
              },
              contained: {
                boxShadow: '0px 1px 2px rgba(0, 0, 0, 0.05)',
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: 'none',
              },
            },
          },
          MuiCard: {
            styleOverrides: {
              root: {
                backgroundImage: 'none',
              },
            },
          },
          MuiListItem: {
            styleOverrides: {
              root: {
                paddingTop: 4,
                paddingBottom: 4,
              },
            },
          },
          MuiTextField: {
            styleOverrides: {
              root: {
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)',
                  },
                },
              },
            },
          },
        },
      }),
    [mode],
  );

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Routes>
            <Route path="/" element={<TranscriptAnalyzer />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </Router>
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

export default App; 