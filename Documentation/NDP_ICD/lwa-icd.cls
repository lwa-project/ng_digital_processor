% lwa-icd.cls - LaTeX document class for LWA Interface Control Documents
%
% Provides common formatting, callout boxes, metadata commands, title page,
% and approval signature blocks for LWA project ICDs.
%
% Usage:
%   \documentclass[11pt,letterpaper]{lwa-icd}
%
%   % Set metadata (override defaults with \renewcommand)
%   \renewcommand{\icdAbbrev}{ASP}
%   \renewcommand{\docVersion}{I}
%   \renewcommand{\docTitle}{Analog Signal Processor (ASP)\\Interface Control Document}
%   ...
%
%   \begin{document}
%   \makeICDtitlepage
%   ...
%   \end{document}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lwa-icd}[2026/02/05 v1.0 LWA Interface Control Document class]

% ============================================================================
% Base class
% ============================================================================

\LoadClass[11pt,letterpaper]{report}

% ============================================================================
% Required packages
% ============================================================================

% Page layout
\RequirePackage[margin=1in,headheight=14pt]{geometry}

% Headers and footers
\RequirePackage{fancyhdr}

% Tables
\RequirePackage{longtable}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{array}

% Callout / annotation boxes
\RequirePackage[most]{tcolorbox}

% Hyperlinks and cross-references
\RequirePackage{hyperref}

% Code listings
\RequirePackage{listings}
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    columns=fullflexible
}

% Graphics and colors
\RequirePackage{graphicx}
\RequirePackage{xcolor}

% Math and units
\RequirePackage{amsmath}
\RequirePackage{siunitx}

% Formatting
\RequirePackage{parskip}
\RequirePackage{enumitem}
\RequirePackage{underscore}

% ============================================================================
% Document metadata
%
% Override these with \renewcommand in your document preamble or metadata.tex.
% ============================================================================

\newcommand{\icdAbbrev}{}                  % e.g., ASP, SHL, NDP
\newcommand{\docVersion}{}                 % e.g., I, F
\newcommand{\docDate}{}                    % e.g., February 2026
\newcommand{\prevVersion}{}                % e.g., H
\newcommand{\prevDate}{}                   % e.g., April 7, 2015
\newcommand{\docAuthors}{}                 % e.g., J.~Craig, J.~Dowell
\newcommand{\docNumber}{}                  % e.g., ASP_ICD
\newcommand{\docTitle}{}                   % Full title with line breaks
\newcommand{\docSubtitle}{Version \docVersion}
\newcommand{\orgName}{Long Wavelength Array Project}
\newcommand{\orgAddress}{%
    University of New Mexico\\
    Department of Physics and Astronomy\\
    Albuquerque, NM 87131
}
\newcommand{\sysEngApproval}{System Engineering}
\newcommand{\sysArchApproval}{System Architecture}

% ============================================================================
% Page style
% ============================================================================

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\icdAbbrev{} ICD}
\fancyhead[R]{Version \docVersion}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% Rename chapters to sections
\renewcommand{\chaptername}{Section}

% ============================================================================
% Hyperref setup (deferred so metadata can be set first)
% ============================================================================

\AtBeginDocument{%
    \hypersetup{
        colorlinks=true,
        linkcolor=blue,
        filecolor=magenta,
        urlcolor=cyan,
        citecolor=blue,
    }%
}

% ============================================================================
% Callout box environments
% ============================================================================

% General note box (gray)
\newtcolorbox{noteBox}[1][]{
    colback=gray!10,
    colframe=gray!70,
    fonttitle=\bfseries,
    title={Note},
    #1
}

% Site-specific configuration note (orange)
\newtcolorbox{siteNote}[1][]{
    colback=orange!10,
    colframe=orange!70,
    fonttitle=\bfseries,
    title={Site-Specific Configuration},
    #1
}

% Generic "New in Version X" box (green)
%   Usage: \begin{newInVersion}{I} ... \end{newInVersion}
\newtcolorbox{newInVersion}[2][]{
    colback=green!10,
    colframe=green!70,
    fonttitle=\bfseries,
    title={New in Version #2},
    #1
}

% Generic callout box with custom title and color
%   Usage: \begin{calloutBox}{Title Text}{color} ... \end{calloutBox}
\newtcolorbox{calloutBox}[3][]{
    colback=#3!10,
    colframe=#3!70,
    fonttitle=\bfseries,
    title={#2},
    #1
}

% Factory command to create project-specific callout box environments.
%   #1 = environment name
%   #2 = title text
%   #3 = color name (e.g., orange, blue, red)
%
%   Example:
%     \newicdcallout{revHNote}{Rev H System Difference}{orange}
%     \newicdcallout{hvacNote}{HVAC Controller Note}{blue}
%
%   Then use:
%     \begin{revHNote} ... \end{revHNote}
\newcommand{\newicdcallout}[3]{%
    \newtcolorbox{#1}[1][]{%
        colback=#3!10,%
        colframe=#3!70,%
        fonttitle=\bfseries,%
        title={#2},%
        ##1%
    }%
}

% ============================================================================
% Inline formatting commands
% ============================================================================

% Inline "New in Version X" label for use in section headings etc.
%   Usage: \section{LOC Command \newInVer{I}}
\newcommand{\newInVer}[1]{%
    \textnormal{\textcolor{green!70!black}{\textbf{[New in Version #1]}}}%
}

% Inline margin note for site-specific differences
\newcommand{\sitemargin}[1]{\marginpar{\small\textcolor{orange}{Site: #1}}}

% Command name formatting (e.g., \cmd{PNG})
\newcommand{\cmd}[1]{\texttt{#1}}

% MIB entry formatting (e.g., \mib{SUMMARY})
\newcommand{\mib}[1]{\texttt{#1}}

% Hex value formatting (e.g., \hex{0A} produces 0x0A)
\newcommand{\hex}[1]{\texttt{0x#1}}

% Register/address formatting
\newcommand{\reg}[1]{\texttt{#1}}

% ============================================================================
% Title page
% ============================================================================

% Approval / signature block (two-column with signature lines)
%   Can be used standalone or as part of \makeICDtitlepage.
\newcommand{\makeapprovalblock}{%
    \begin{tabular}{p{0.45\textwidth}p{0.45\textwidth}}
        \textbf{Approved:} & \textbf{Approved:} \\[1.5cm]
        \hrulefill & \hrulefill \\
        \sysEngApproval & \sysArchApproval \\[0.5cm]
        Date: \hrulefill & Date: \hrulefill \\
    \end{tabular}%
}

% Full ICD title page.  Expects a figures/logo image to be available.
\newcommand{\makeICDtitlepage}{%
    \begin{titlepage}
        \centering
        \vspace*{1cm}
        \begin{center}
            \includegraphics[width=0.3\textwidth]{figures/logo}
        \end{center}
        \vspace{1.5cm}
        {\LARGE\bfseries \docTitle \par}
        \vspace{0.5cm}
        {\Large \docSubtitle \par}
        \vspace{2cm}
        {\large \docAuthors \par}
        \vspace{1cm}
        {\large \docDate \par}
        \vfill
        \makeapprovalblock
        \vspace{1cm}
        {\normalsize \orgName \par}
    \end{titlepage}%
}

% ============================================================================
% Front matter helpers
% ============================================================================

% Change record chapter header (unnumbered, added to TOC)
\newcommand{\changerecordheader}{%
    \chapter*{Change Record}%
    \addcontentsline{toc}{chapter}{Change Record}%
}

% Standard TOC + List of Tables sequence
\newcommand{\makeICDtoc}{%
    \tableofcontents
    \newpage
    \listoftables
    \addcontentsline{toc}{chapter}{List of Tables}
    \newpage
}

\endinput
