# Training Images Directory

This directory contains training images for face recognition. The structure is organized as follows:

## Directory Structure

- `team_members/` - Contains training images for team members
  - Each team member should have their own subdirectory named with their username
  - Example: `team_members/john_doe/`

## How to Add Training Images

1. Create a subdirectory for each team member using their username
2. Add multiple face images of the team member in their directory
3. Image requirements:
   - Format: JPG or PNG
   - Clear face shots
   - Different angles and expressions recommended
   - Minimum size: 640x480 pixels
   - Maximum size: 1920x1080 pixels

## Example Structure

```
training_images/
├── team_members/
│   ├── john_doe/
│   │   ├── front.jpg
│   │   ├── side.jpg
│   │   └── smile.jpg
│   └── jane_smith/
│       ├── front.jpg
│       ├── side.jpg
│       └── smile.jpg
```

## Best Practices

1. Use clear, well-lit images
2. Include different angles of the face
3. Include different expressions
4. Use consistent image quality
5. Name files descriptively
6. Keep file sizes reasonable (under 1MB per image)

## Notes

- The system will automatically detect and use these images for training
- More training images generally lead to better recognition accuracy
- Images should be of good quality and clearly show the face
- Avoid using blurry or dark images 