
const { Router } = require('express');
const {getResumeSummary} = require('../controllers/resume-summary');

const router = Router();


router.get('/',getResumeSummary);

module.exports = router;