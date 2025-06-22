import React, { useState } from 'react';
import { 
  Box, Button, Container, Typography, Paper, CircularProgress, 
  FormControl, InputLabel, MenuItem, Select, Snackbar, Alert, TextField, IconButton,
  Tooltip, useTheme, alpha, Divider, FormControlLabel, Switch
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import SummarizeIcon from '@mui/icons-material/Summarize';
import SettingsIcon from '@mui/icons-material/Settings';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import CloudIcon from '@mui/icons-material/Cloud';
import ComputerIcon from '@mui/icons-material/Computer';
import ThemeToggle from './ThemeToggle';

const TranscriptAnalyzer = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  
  // State variables
  const [file, setFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isCacheClearing, setIsCacheClearing] = useState(false);
  const [modelType, setModelType] = useState('local'); // 'local' or 'openai'
  const [model, setModel] = useState('mistral');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [openaiModel, setOpenaiModel] = useState('gpt-3.5-turbo');
  const [error, setError] = useState('');
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');
  const [notificationType, setNotificationType] = useState('info');
  const [showSettings, setShowSettings] = useState(false);

  // Handle file upload
  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      // Check file type
      if (!uploadedFile.name.endsWith('.txt') && !uploadedFile.name.endsWith('.md')) {
        showAlert('Please upload a .txt or .md file', 'error');
        return;
      }

      setFile(uploadedFile);
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target.result);
      };
      reader.readAsText(uploadedFile);
      showAlert('File uploaded successfully', 'success');
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    if (!fileContent) {
      showAlert('Please upload a transcript file first', 'error');
      return;
    }

    if (modelType === 'openai' && !openaiApiKey) {
      showAlert('Please enter an OpenAI API key', 'error');
      return;
    }

    setIsLoading(true);
    setError('');
    
    // Show longer processing time message for phi model
    if (model === 'phi' && modelType === 'local') {
      showAlert('Using phi model - this may take longer to process', 'info');
    }

    try {
      // Set longer timeout for phi model
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 
        modelType === 'openai' ? 120000 : (model === 'phi' ? 300000 : 120000)); // 2 min for OpenAI, 5 min for phi, 2 min for others
      
      // Prepare request body based on model type
      let requestBody = {
        text: fileContent,
        max_chunk_size: 1500,
        use_cache: true
      };
      
      if (modelType === 'local') {
        requestBody.model = model;
      } else if (modelType === 'openai') {
        requestBody.llm_type = 'openai';
        requestBody.api_url = 'https://api.openai.com/v1/chat/completions';
        requestBody.api_key = openaiApiKey;
        requestBody.model = openaiModel;
      }
      
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      showAlert('Analysis completed successfully', 'success');
      
      // Navigate to results page with the data
      navigate('/results', { state: { results: data } });
    } catch (err) {
      setError(err.message);
      showAlert(`Error: ${err.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle cache clearing
  const handleClearCache = async () => {
    setIsCacheClearing(true);
    try {
      const response = await fetch('http://localhost:8000/cache', {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const result = await response.json();
      showAlert(`Cache cleared: ${result.message}`, 'success');
    } catch (err) {
      showAlert(`Error clearing cache: ${err.message}`, 'error');
    } finally {
      setIsCacheClearing(false);
    }
  };

  // Show notification
  const showAlert = (message, type) => {
    setNotificationMessage(message);
    setNotificationType(type);
    setShowNotification(true);
  };

  // Close notification
  const handleCloseNotification = () => {
    setShowNotification(false);
  };

  return (
    <Container maxWidth="md" sx={{ py: 5 }}>
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center',
        mb: 4,
        position: 'relative'
      }}>
        <Box sx={{ position: 'absolute', top: 0, right: 0 }}>
          <ThemeToggle />
        </Box>
        <Typography 
          variant="h3" 
          component="h1" 
          gutterBottom 
          align="center"
          sx={{ 
            fontWeight: 700, 
            color: theme.palette.text.primary,
            fontSize: { xs: '2rem', sm: '2.5rem' }
          }}
        >
          Meeting Transcript Analyzer
        </Typography>
        <Typography 
          variant="subtitle1" 
          align="center" 
          color="text.secondary"
          sx={{ maxWidth: '600px', mb: 2 }}
        >
          Upload your meeting transcript to extract summaries, decisions, and action items using AI
        </Typography>
      </Box>
      
      <Paper 
        elevation={0} 
        sx={{ 
          p: 4, 
          mb: 4, 
          borderRadius: 3,
          backgroundColor: alpha(theme.palette.background.paper, 0.8),
          backdropFilter: 'blur(10px)',
          border: '1px solid',
          borderColor: alpha(theme.palette.divider, 0.1)
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>Upload Transcript</Typography>
            <Tooltip title="Settings">
              <IconButton 
                size="small" 
                onClick={() => setShowSettings(!showSettings)}
                sx={{ 
                  backgroundColor: showSettings ? alpha(theme.palette.primary.main, 0.1) : 'transparent',
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.1)
                  }
                }}
              >
                <SettingsIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Box sx={{ 
            border: '2px dashed',
            borderColor: alpha(theme.palette.primary.main, 0.3),
            borderRadius: 2,
            p: 3,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            backgroundColor: alpha(theme.palette.primary.main, 0.03),
            transition: 'all 0.2s ease',
            '&:hover': {
              borderColor: theme.palette.primary.main,
              backgroundColor: alpha(theme.palette.primary.main, 0.05)
            }
          }}>
            <Box sx={{ mb: 2 }}>
              <UploadFileIcon sx={{ fontSize: 48, color: alpha(theme.palette.primary.main, 0.7) }} />
            </Box>
            <Typography variant="body1" sx={{ mb: 1 }}>
              {file ? `Selected: ${file.name}` : 'Drag and drop your file here or click to browse'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Accepts .txt or .md files
            </Typography>
            <Button
              variant="outlined"
              component="label"
              sx={{ 
                borderRadius: '20px',
                px: 3
              }}
            >
              Select File
              <input
                type="file"
                hidden
                accept=".txt,.md"
                onChange={handleFileUpload}
              />
            </Button>
          </Box>
          
          {showSettings && (
            <Box sx={{ 
              p: 2, 
              borderRadius: 2,
              backgroundColor: alpha(theme.palette.background.default, 0.5),
              border: '1px solid',
              borderColor: alpha(theme.palette.divider, 0.1)
            }}>
              <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                AI Model Settings
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <Button 
                  variant={modelType === 'local' ? 'contained' : 'outlined'}
                  size="small"
                  startIcon={<ComputerIcon />}
                  onClick={() => setModelType('local')}
                  sx={{ borderRadius: 2, flex: 1 }}
                >
                  Local (Ollama)
                </Button>
                <Button 
                  variant={modelType === 'openai' ? 'contained' : 'outlined'}
                  size="small"
                  startIcon={<CloudIcon />}
                  onClick={() => setModelType('openai')}
                  sx={{ borderRadius: 2, flex: 1 }}
                >
                  OpenAI API
                </Button>
              </Box>
              
              {modelType === 'local' ? (
                <>
                  <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                    <InputLabel id="model-select-label">Local Model</InputLabel>
                    <Select
                      labelId="model-select-label"
                      id="model-select"
                      value={model}
                      label="Local Model"
                      onChange={(e) => setModel(e.target.value)}
                      sx={{ borderRadius: 2 }}
                    >
                      <MenuItem value="mistral">Mistral</MenuItem>
                      <MenuItem value="llama3.2">Llama 3.2</MenuItem>
                      <MenuItem value="gemma">Gemma</MenuItem>
                      <MenuItem value="phi">Phi</MenuItem>
                    </Select>
                  </FormControl>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <InfoOutlinedIcon sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">
                      Mistral requires ~4.8GB RAM, Llama 3.2 requires ~8GB RAM, Phi may take longer to process
                    </Typography>
                  </Box>
                </>
              ) : (
                <>
                  <TextField
                    label="OpenAI API Key"
                    type="password"
                    size="small"
                    fullWidth
                    value={openaiApiKey}
                    onChange={(e) => setOpenaiApiKey(e.target.value)}
                    placeholder="sk-..."
                    sx={{ mb: 2, borderRadius: 2 }}
                  />
                  <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                    <InputLabel id="openai-model-select-label">OpenAI Model</InputLabel>
                    <Select
                      labelId="openai-model-select-label"
                      id="openai-model-select"
                      value={openaiModel}
                      label="OpenAI Model"
                      onChange={(e) => setOpenaiModel(e.target.value)}
                      sx={{ borderRadius: 2 }}
                    >
                      <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo</MenuItem>
                      <MenuItem value="gpt-4">GPT-4</MenuItem>
                    </Select>
                  </FormControl>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <InfoOutlinedIcon sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">
                      GPT-3.5 is faster and cheaper, GPT-4 provides higher quality results
                    </Typography>
                  </Box>
                </>
              )}
              
              <Divider sx={{ my: 2 }} />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Cache Management
                </Typography>
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  startIcon={isCacheClearing ? <CircularProgress size={16} color="error" /> : <DeleteSweepIcon />}
                  onClick={handleClearCache}
                  disabled={isCacheClearing}
                  sx={{ borderRadius: 2 }}
                >
                  {isCacheClearing ? 'Clearing...' : 'Clear Cache'}
                </Button>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                Clearing the cache will remove all saved analysis results
              </Typography>
            </Box>
          )}
          
          {fileContent && (
            <TextField
              label="Transcript Preview"
              multiline
              rows={3}
              value={fileContent.length > 200 ? fileContent.substring(0, 200) + '...' : fileContent}
              variant="outlined"
              fullWidth
              InputProps={{ 
                readOnly: true,
                sx: { 
                  borderRadius: 2,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem'
                }
              }}
            />
          )}
          
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={isLoading || !fileContent}
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SummarizeIcon />}
            sx={{ 
              mt: 1, 
              py: 1.2,
              borderRadius: 2,
              fontWeight: 600,
              boxShadow: 2,
              alignSelf: 'flex-start'
            }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Transcript'}
          </Button>
        </Box>
      </Paper>
      
      {/* Error message */}
      {error && (
        <Paper 
          elevation={0} 
          sx={{ 
            p: 3, 
            mt: 3, 
            borderRadius: 2,
            backgroundColor: alpha(theme.palette.error.light, 0.1),
            border: '1px solid',
            borderColor: alpha(theme.palette.error.main, 0.2)
          }}
        >
          <Typography color="error" variant="body2" sx={{ fontWeight: 500 }}>
            {error}
          </Typography>
        </Paper>
      )}
      
      {/* Notification */}
      <Snackbar 
        open={showNotification} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notificationType} 
          sx={{ width: '100%', borderRadius: 2 }}
          elevation={6}
        >
          {notificationMessage}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default TranscriptAnalyzer;