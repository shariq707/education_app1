![Home1](https://github.com/user-attachments/assets/2fa7f7d9-d6b8-4cd8-add0-c7fbdd414b22)
![Admin1](https://github.com/user-attachments/assets/93f35e3b-74e5-4676-9b5c-d94c22fdd7f3)
![TeacherDashboard](https://github.com/user-attachments/assets/89a6b1e8-0e2d-402f-8826-5df006dbc519)
![Studentdashboard](https://github.com/user-attachments/assets/dd5703bd-1853-496a-ae07-1926dae58052)
![Analystdashboard](https://github.com/user-attachments/assets/a50c992f-1372-402d-81b3-96df54d1101b)



# Education Management System

A comprehensive web-based education management system built with Flask and MongoDB, designed to facilitate learning and teaching processes with features for students, teachers, and administrators.

## Features

### Student Features
- Course enrollment and management
- Assignment submission and tracking
- Progress monitoring and performance analytics
- Access to course materials and announcements
- View grades and feedback

### Teacher Features
- Course creation and management
- Assignment creation and grading
- Student performance analytics
- Announcement system
- Class management

### Administrator Features
- User management (students, teachers, analysts)
- System analytics and reporting
- Data ingestion and processing
- Notification management
- System monitoring

### Analyst Features
- Data visualization and analytics
- Predictive modeling
- Report generation
- Dataset management

## Prerequisites

- Python 3.8+
- MongoDB 4.4+
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd education_app1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with the following variables:
   ```
   MONGO_URI=mongodb://localhost:27017/education_app
   SECRET_KEY=your-secret-key-here
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_USERNAME=your-email@gmail.com
   MAIL_PASSWORD=your-email-password
   MAIL_USE_TLS=true
   ```

5. **Start MongoDB service**
   Make sure MongoDB is running on your system.

## Running the Application

1. **Initialize the database**
   The application will create necessary collections on first run.

2. **Start the development server**
   ```bash
   python app.py
   ```

3. **Access the application**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
education_app1/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (create this file)
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── csv/                 # Sample CSV data files
├── ingestion/           # Data ingestion utilities
└── README.md            # This file
```

## API Endpoints

The application provides various API endpoints for data operations, including:

- Authentication (login, signup, password reset)
- User management
- Course management
- Assignment submission and grading
- Analytics and reporting
- Data ingestion

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the repository or contact the development team.

---

*This project was developed as part of an educational initiative to enhance learning management systems through technology.*
