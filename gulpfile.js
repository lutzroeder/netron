var gulp = require("gulp");
var sourcemaps = require('gulp-sourcemaps');
var typescript = require('gulp-typescript');
var uglify = require('gulp-uglify');
var open = require('gulp-open');

gulp.task("build", [ "build:page", "build:code" ], function() {
});

gulp.task("build:code", function() {
    return gulp.src("./src/*.ts")
        .pipe(sourcemaps.init())
        .pipe(typescript({ target: "ES5", out: "netron.js" }))
        .once("error", function() { this.once("finish", () => process.exit(1)) })
        .pipe(uglify())
        .pipe(sourcemaps.write("."))
        .pipe(gulp.dest("./build"));
});

gulp.task("build:page", function() {
    return gulp.src([ "./samples/index.html" ])
        .pipe(gulp.dest("./build"))
});

gulp.task("default", [ "build" ], function() {
    return gulp.src("./build/index.html")
        .pipe(open());
});