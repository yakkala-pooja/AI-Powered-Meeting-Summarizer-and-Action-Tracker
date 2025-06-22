import React, { useState } from 'react';
import { 
  Box, Button, Container, Typography, CircularProgress,
  List, ListItem, ListItemText, Snackbar, Alert, IconButton,
  Card, CardContent, Chip, Tooltip, useTheme, alpha,
  Dialog, DialogTitle, DialogContent, DialogActions, DialogContentText,
  TextField
} from '@mui/material';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import { saveAs } from 'file-saver';
import { useNavigate, useLocation } from 'react-router-dom';
import SummarizeIcon from '@mui/icons-material/Summarize';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AssignmentTurnedInIcon from '@mui/icons-material/AssignmentTurnedIn';
import MarkdownIcon from '@mui/icons-material/Code';
import HtmlIcon from '@mui/icons-material/Html';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import EmailIcon from '@mui/icons-material/Email';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ThemeToggle from './ThemeToggle';

const ResultsPage = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const { results } = location.state || {};
  
  // State variables
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');
  const [notificationType, setNotificationType] = useState('info');
  const [emailDialogOpen, setEmailDialogOpen] = useState(false);
  const [email, setEmail] = useState('');
  const [emailSending, setEmailSending] = useState(false);
  const [exportingPdf, setExportingPdf] = useState(false);

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

  // Open email dialog
  const handleOpenEmailDialog = () => {
    setEmailDialogOpen(true);
  };

  // Close email dialog
  const handleCloseEmailDialog = () => {
    setEmailDialogOpen(false);
  };

  // Send email with results
  const handleSendEmail = async () => {
    if (!email || !results) return;

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      showAlert('Please enter a valid email address', 'error');
      return;
    }

    setEmailSending(true);

    try {
      // Create email content
      const emailContent = {
        to: email,
        subject: 'Meeting Analysis Results',
        content: {
          summary: results.results.summary,
          decisions: results.results.decisions,
          action_items: results.results.action_items,
          metadata: {
            model: results.model_used,
            processing_time: results.processing_time,
            fallback_used: results.fallback_used
          }
        }
      };

      // Send to backend
      const response = await fetch('http://localhost:8000/send-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(emailContent),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      showAlert('Email sent successfully!', 'success');
      handleCloseEmailDialog();
    } catch (err) {
      showAlert(`Error sending email: ${err.message}`, 'error');
    } finally {
      setEmailSending(false);
    }
  };

  // Export as Markdown
  const exportMarkdown = () => {
    if (!results) return;

    let markdown = `# Meeting Analysis Results\n\n`;
    
    // Add summary
    markdown += `## Summary\n\n`;
    results.results.summary.forEach(item => {
      markdown += `- ${item}\n`;
    });
    
    // Add decisions
    markdown += `\n## Decisions\n\n`;
    if (results.results.decisions && results.results.decisions.length > 0) {
      results.results.decisions.forEach(item => {
        markdown += `- ${item}\n`;
      });
    } else {
      markdown += `- None identified\n`;
    }
    
    // Add action items
    markdown += `\n## Action Items\n\n`;
    if (results.results.action_items && results.results.action_items.length > 0) {
      results.results.action_items.forEach(item => {
        markdown += `- ${item}\n`;
      });
    } else {
      markdown += `- None identified\n`;
    }
    
    // Add metadata
    markdown += `\n## Metadata\n\n`;
    markdown += `- Model used: ${results.model_used}\n`;
    markdown += `- Processing time: ${results.processing_time.toFixed(2)} seconds\n`;
    markdown += `- Fallback used: ${results.fallback_used ? 'Yes' : 'No'}\n`;
    
    // Create and download file
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
    saveAs(blob, 'meeting_analysis.md');
    showAlert('Exported as Markdown', 'success');
  };

  // Export as HTML
  const exportHTML = () => {
    if (!results) return;

    let html = `<!DOCTYPE html>
<html>
<head>
  <title>Meeting Analysis Results</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 0; padding: 40px; color: #37352f; line-height: 1.5; }
    .container { max-width: 800px; margin: 0 auto; }
    h1 { font-size: 2.5em; font-weight: 600; margin-bottom: 1em; }
    h2 { font-size: 1.5em; font-weight: 500; margin-top: 1.5em; color: #37352f; }
    ul { margin-bottom: 20px; }
    li { margin-bottom: 8px; }
    .section { margin-bottom: 2em; padding: 1.5em; border-radius: 8px; background-color: #f7f6f3; }
    .metadata { background-color: #eae9e5; padding: 20px; border-radius: 8px; margin-top: 30px; }
    .tag { display: inline-block; background-color: #e9e8e4; color: #37352f; padding: 4px 10px; border-radius: 4px; margin-right: 8px; font-size: 0.9em; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Meeting Analysis Results</h1>
    
    <div class="section">
      <h2>Summary</h2>
      <ul>
        ${results.results.summary.map(item => `<li>${item}</li>`).join('')}
      </ul>
    </div>
    
    <div class="section">
      <h2>Decisions</h2>
      <ul>
        ${results.results.decisions && results.results.decisions.length > 0 
          ? results.results.decisions.map(item => `<li>${item}</li>`).join('') 
          : '<li>None identified</li>'}
      </ul>
    </div>
    
    <div class="section">
      <h2>Action Items</h2>
      <ul>
        ${results.results.action_items && results.results.action_items.length > 0 
          ? results.results.action_items.map(item => `<li>${item}</li>`).join('') 
          : '<li>None identified</li>'}
      </ul>
    </div>
    
    <div class="metadata">
      <h2>Metadata</h2>
      <p><span class="tag">Model</span> ${results.model_used}</p>
      <p><span class="tag">Processing time</span> ${results.processing_time.toFixed(2)} seconds</p>
      <p><span class="tag">Fallback used</span> ${results.fallback_used ? 'Yes' : 'No'}</p>
    </div>
  </div>
</body>
</html>`;

    // Create and download file
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    saveAs(blob, 'meeting_analysis.html');
    showAlert('Exported as HTML', 'success');
  };

  // Export as PDF
  const exportPDF = async () => {
    if (!results || !document.getElementById('results-container')) return;

    try {
      setExportingPdf(true);
      const element = document.getElementById('results-container');
      const canvas = await html2canvas(element, { scale: 2 });
      const imgData = canvas.toDataURL('image/png');
      
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      
      pdf.addImage(imgData, 'PNG', imgX, 0, imgWidth * ratio, imgHeight * ratio);
      pdf.save('meeting_analysis.pdf');
      showAlert('Exported as PDF', 'success');
    } catch (err) {
      showAlert(`Error exporting PDF: ${err.message}`, 'error');
    } finally {
      setExportingPdf(false);
    }
  };

  // Render a section with items
  const renderSection = (title, items, icon) => {
    return (
      <Card 
        elevation={0} 
        sx={{ 
          mb: 3, 
          border: '1px solid',
          borderColor: alpha(theme.palette.divider, 0.1),
          borderRadius: 2,
          overflow: 'visible'
        }}
      >
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 2,
            pb: 1,
            borderBottom: '1px solid',
            borderColor: alpha(theme.palette.divider, 0.1)
          }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              borderRadius: '50%',
              width: 32,
              height: 32,
              backgroundColor: alpha(theme.palette.primary.main, 0.1),
              mr: 1.5
            }}>
              {icon}
            </Box>
            <Typography variant="h6" sx={{ fontWeight: 600, color: theme.palette.text.primary }}>
              {title}
            </Typography>
          </Box>
          
          <List sx={{ pl: 1 }}>
            {items && items.length > 0 ? (
              items.map((item, index) => (
                <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
                  <ListItemText 
                    primary={
                      <Typography variant="body1" sx={{ fontWeight: 400, color: theme.palette.text.primary }}>
                        {item}
                      </Typography>
                    } 
                  />
                </ListItem>
              ))
            ) : (
              <ListItem sx={{ py: 0.5, px: 0 }}>
                <ListItemText 
                  primary={
                    <Typography variant="body1" sx={{ fontStyle: 'italic', color: theme.palette.text.secondary }}>
                      None identified
                    </Typography>
                  } 
                />
              </ListItem>
            )}
          </List>
        </CardContent>
      </Card>
    );
  };

  // Go back to the main page
  const handleGoBack = () => {
    navigate('/');
  };

  // If no results are available, redirect back to home
  if (!results) {
    return (
      <Container maxWidth="md" sx={{ py: 5, textAlign: 'center' }}>
        <Box sx={{ position: 'absolute', top: 20, right: 20 }}>
          <ThemeToggle />
        </Box>
        <Typography variant="h5" sx={{ mb: 3 }}>No analysis results available</Typography>
        <Button 
          variant="contained"
          startIcon={<ArrowBackIcon />}
          onClick={handleGoBack}
        >
          Go Back to Analyzer
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 5, position: 'relative' }}>
      <Box sx={{ position: 'absolute', top: 20, right: 20 }}>
        <ThemeToggle />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Button
          variant="outlined"
          startIcon={<ArrowBackIcon />}
          onClick={handleGoBack}
          sx={{ mb: 3 }}
        >
          Back to Analyzer
        </Button>
      </Box>
      
      {/* Results Section */}
      <Box 
        id="results-container"
        sx={{ 
          mb: 4,
          backgroundColor: alpha(theme.palette.background.paper, 0.8),
          backdropFilter: 'blur(10px)',
          borderRadius: 3,
          p: 4,
          border: '1px solid',
          borderColor: alpha(theme.palette.divider, 0.1)
        }}
      >
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          mb: 3,
          pb: 2,
          borderBottom: '1px solid',
          borderColor: alpha(theme.palette.divider, 0.1)
        }}>
          <Typography 
            variant="h5" 
            sx={{ 
              fontWeight: 700,
              color: theme.palette.text.primary
            }}
          >
            Analysis Results
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Email Results">
              <IconButton 
                size="small" 
                onClick={handleOpenEmailDialog}
                sx={{ 
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.2)
                  }
                }}
              >
                <EmailIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as Markdown">
              <IconButton 
                size="small" 
                onClick={exportMarkdown}
                sx={{ 
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.2)
                  }
                }}
              >
                <MarkdownIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as HTML">
              <IconButton 
                size="small" 
                onClick={exportHTML}
                sx={{ 
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.2)
                  }
                }}
              >
                <HtmlIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as PDF">
              <IconButton 
                size="small" 
                onClick={exportPDF}
                disabled={exportingPdf}
                sx={{ 
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.2)
                  }
                }}
              >
                {exportingPdf ? <CircularProgress size={16} /> : <PictureAsPdfIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        {renderSection('Summary', results.results.summary, <SummarizeIcon color="primary" fontSize="small" />)}
        {renderSection('Decisions', results.results.decisions, <CheckCircleIcon color="primary" fontSize="small" />)}
        {renderSection('Action Items', results.results.action_items, <AssignmentTurnedInIcon color="primary" fontSize="small" />)}
        
        <Box sx={{ 
          display: 'flex',
          flexWrap: 'wrap',
          gap: 1,
          mt: 3,
          pt: 2,
          borderTop: '1px solid',
          borderColor: alpha(theme.palette.divider, 0.1)
        }}>
          <Chip 
            label={`Model: ${results.model_used}`} 
            size="small" 
            sx={{ 
              backgroundColor: alpha(theme.palette.primary.main, 0.1),
              color: theme.palette.primary.main,
              fontWeight: 500
            }}
          />
          <Chip 
            label={`Processing time: ${results.processing_time.toFixed(2)}s`} 
            size="small" 
            sx={{ 
              backgroundColor: alpha(theme.palette.info.main, 0.1),
              color: theme.palette.info.main,
              fontWeight: 500
            }}
          />
          <Chip 
            label={`Fallback used: ${results.fallback_used ? 'Yes' : 'No'}`} 
            size="small" 
            sx={{ 
              backgroundColor: results.fallback_used 
                ? alpha(theme.palette.warning.main, 0.1)
                : alpha(theme.palette.success.main, 0.1),
              color: results.fallback_used
                ? theme.palette.warning.main
                : theme.palette.success.main,
              fontWeight: 500
            }}
          />
        </Box>
      </Box>
      
      {/* Email Dialog */}
      <Dialog open={emailDialogOpen} onClose={handleCloseEmailDialog}>
        <DialogTitle>Email Meeting Results</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Enter your email address to receive the meeting summary, decisions, and action items.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            id="email"
            label="Email Address"
            type="email"
            fullWidth
            variant="outlined"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseEmailDialog}>Cancel</Button>
          <Button 
            onClick={handleSendEmail} 
            variant="contained" 
            disabled={emailSending || !email}
            startIcon={emailSending ? <CircularProgress size={16} color="inherit" /> : null}
          >
            {emailSending ? 'Sending...' : 'Send'}
          </Button>
        </DialogActions>
      </Dialog>
      
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

export default ResultsPage;