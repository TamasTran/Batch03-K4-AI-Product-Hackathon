import React, { useState } from 'react';

const DATASETS = [
    {
        id: 1,
        name: 'UIT-VSFC Vietnamese Sentiment',
        source: 'Hugging Face',
        relevance: 94,
        quality: { status: 'green', title: 'Dùng được ngay', note: 'Dữ liệu sạch, đầy đủ gán nhãn' },
        license: { status: 'green', title: 'CC-BY', note: 'Được phép dùng & chỉnh sửa với ghi công' },
        url: 'https://huggingface.co/datasets/uit-vsc/vsfc'
    },
    {
        id: 2,
        name: 'Vietnamese E-commerce Reviews',
        source: 'Kaggle',
        relevance: 81,
        quality: { status: 'yellow', title: 'Cần kiểm tra thủ công', note: 'Thiếu 12% nhãn, cần làm sạch' },
        license: { status: 'green', title: 'CC-BY-SA', note: 'Được phép dùng với cùng điều khoản' },
        url: 'https://www.kaggle.com/datasets/khanhnamle1728/vietnamese-ecommerce'
    },
    {
        id: 3,
        name: 'Shopee VN Comments Dump',
        source: 'Nội bộ',
        relevance: 76,
        quality: { status: 'yellow', title: 'Cần kiểm tra thủ công', note: 'Chưa rõ nguồn thu thập, cần xác thực' },
        license: { status: 'yellow', title: 'Chưa xác định', note: 'Cần liên hệ chủ sở hữu về điều khoản' },
        url: '#'
    }
];

const EXAMPLES = [
    'Dữ liệu bình luận tiếng Việt có gán nhãn cảm xúc',
    'Review sản phẩm E-commerce Việt Nam đầy đủ',
    'Danh sách khách hàng churn với lý do từ chối'
];

export default function DataScout() {
    const [screen, setScreen] = useState('onboarding');
    const [query, setQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleStartClick = () => setScreen('search');

    const handleSearch = () => {
        if (!query.trim()) {
            alert('Vui lòng nhập câu hỏi trước khi tìm kiếm.');
            return;
        }
        setIsLoading(true);
        setTimeout(() => {
            setIsLoading(false);
            setScreen('results');
        }, 1000);
    };

    const handleExampleClick = (example) => {
        setQuery(example);
    };

    const handleBack = () => {
        setQuery('');
        setScreen('search');
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.container}>
                <header style={styles.header}>
                    <div style={styles.logo}>DataScout</div>
                    <div style={styles.tagline}>Biết dataset có dùng được — trước khi tải về</div>
                </header>

                {screen === 'onboarding' && (
                    <OnboardingScreen onStart={handleStartClick} />
                )}

                {screen === 'search' && (
                    <SearchScreen
                        query={query}
                        onQueryChange={setQuery}
                        onSearch={handleSearch}
                        onExampleClick={handleExampleClick}
                        isLoading={isLoading}
                    />
                )}

                {screen === 'results' && (
                    <ResultsScreen onBack={handleBack} isLoading={isLoading} />
                )}
            </div>

            <footer style={styles.footer}>
                <div style={styles.footerText}>Biết Sớm, Tải Thông Minh</div>
            </footer>
        </div>
    );
}

function OnboardingScreen({ onStart }) {
    const steps = [
        { num: 1, title: 'Mô tả bài toán', desc: 'Viết một câu tiếng Việt mô tả dataset bạn cần' },
        { num: 2, title: 'Xem kết quả', desc: 'Nhận kết quả kèm điểm phù hợp & cờ chất lượng' },
        { num: 3, title: 'Đọc cờ', desc: 'Kiểm tra trước khi quyết định tải về' },
        { num: 4, title: 'Lấy nguồn', desc: 'Truy cập trực tiếp kèm trích dẫn chuẩn' }
    ];

    return (
        <div style={styles.onboarding}>
            <div style={styles.stepsGrid}>
                {steps.map(step => (
                    <div key={step.num} style={styles.stepCard}>
                        <div style={styles.stepNumber}>{step.num}</div>
                        <div style={styles.stepTitle}>{step.title}</div>
                        <div style={styles.stepDesc}>{step.desc}</div>
                    </div>
                ))}
            </div>

            <button onClick={onStart} style={styles.primaryButton}>
                Bắt đầu
            </button>
        </div>
    );
}

function SearchScreen({ query, onQueryChange, onSearch, onExampleClick, isLoading }) {
    return (
        <div style={styles.searchSection}>
            {isLoading && (
                <div style={styles.loadingContainer}>
                    <div style={styles.spinner}></div>
                    <p>Đang tìm kiếm dataset phù hợp...</p>
                </div>
            )}

            {!isLoading && (
                <>
                    <div style={styles.inputWrapper}>
                        <label style={styles.label}>Mô tả bài toán của bạn</label>
                        <textarea
                            style={styles.textarea}
                            placeholder="Ví dụ: dữ liệu bình luận tiếng Việt có gán nhãn cảm xúc"
                            value={query}
                            onChange={(e) => onQueryChange(e.target.value)}
                        />
                    </div>

                    <div style={styles.examplesSection}>
                        <div style={styles.examplesLabel}>Ví dụ nhanh:</div>
                        <div style={styles.chipGroup}>
                            {EXAMPLES.map((example, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => onExampleClick(example)}
                                    style={{
                                        ...styles.chip,
                                        ...(query === example ? styles.chipActive : {})
                                    }}
                                >
                                    {example}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div style={styles.buttonGroup}>
                        <button onClick={onSearch} style={styles.primaryButton}>
                            Tìm dataset
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

function ResultsScreen({ onBack, isLoading }) {
    return (
        <div style={styles.resultsSection}>
            {isLoading && (
                <div style={styles.loadingContainer}>
                    <div style={styles.spinner}></div>
                    <p>Đang tải kết quả...</p>
                </div>
            )}

            {!isLoading && (
                <>
                    <div style={styles.resultsHeader}>
                        <h2 style={styles.resultsTitle}>Kết quả tìm kiếm</h2>
                        <p style={styles.resultsSubtitle}>3 dataset phù hợp nhất với yêu cầu của bạn</p>
                    </div>

                    <div style={styles.cardsGrid}>
                        {DATASETS.map(dataset => (
                            <DatasetCard key={dataset.id} dataset={dataset} />
                        ))}
                    </div>

                    <div style={styles.backButtonContainer}>
                        <button onClick={onBack} style={styles.secondaryButton}>
                            ← Tìm kiếm lại
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

function DatasetCard({ dataset }) {
    return (
        <div style={styles.card}>
            <div style={styles.cardHeader}>
                <div style={styles.datasetName}>{dataset.name}</div>
                <div style={styles.datasetSource}>{dataset.source}</div>
            </div>

            <div style={styles.cardSection}>
                <div style={styles.sectionLabel}>Điểm Phù Hợp</div>
                <div style={styles.relevanceScore}>
                    <div style={styles.scoreBadge}>{dataset.relevance}</div>
                    <div style={styles.scoreBar}>
                        <div style={{
                            ...styles.scoreFill,
                            width: `${dataset.relevance}%`
                        }} />
                    </div>
                </div>
            </div>

            <div style={styles.cardSection}>
                <div style={styles.sectionLabel}>Chất Lượng Dữ Liệu</div>
                <Flag flag={dataset.quality} />
            </div>

            <div style={styles.cardSection}>
                <div style={styles.sectionLabel}>Điều Khoản Sử Dụng</div>
                <Flag flag={dataset.license} />
            </div>

            <a href={dataset.url} target="_blank" rel="noopener noreferrer" style={styles.cardLink}>
                Tới nguồn →
            </a>
        </div>
    );
}

function Flag({ flag }) {
    const isGreen = flag.status === 'green';
    const iconBg = isGreen ? '#5FB89A' : '#D98E5F';
    const icon = isGreen ? '✓' : '⚠';

    return (
        <div style={styles.flag}>
            <div style={{
                ...styles.flagIcon,
                backgroundColor: iconBg,
            }}>
                {icon}
            </div>
            <div style={styles.flagContent}>
                <div style={styles.flagTitle}>{flag.title}</div>
                <div style={styles.flagNote}>{flag.note}</div>
            </div>
        </div>
    );
}

const styles = {
    wrapper: {
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        background: 'linear-gradient(135deg, #1A2620 0%, #0E1512 100%)',
        color: '#F4F3EF',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        padding: '2rem 1rem',
    },
    container: {
        maxWidth: '1024px',
        margin: '0 auto',
        width: '100%',
        flex: 1,
    },
    header: {
        textAlign: 'center',
        marginBottom: '3rem',
    },
    logo: {
        fontSize: '2.5rem',
        fontWeight: 800,
        marginBottom: '0.5rem',
        letterSpacing: '-1px',
    },
    tagline: {
        fontSize: '1rem',
        color: '#9FB0A5',
        letterSpacing: '0.05em',
    },
    onboarding: {
        display: 'flex',
        flexDirection: 'column',
        gap: '2rem',
        alignItems: 'center',
    },
    stepsGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1.5rem',
        width: '100%',
        marginBottom: '2rem',
    },
    stepCard: {
        background: 'rgba(15, 21, 18, 0.6)',
        border: '1px solid rgba(159, 176, 165, 0.2)',
        borderRadius: '12px',
        padding: '1.5rem',
        textAlign: 'center',
    },
    stepNumber: {
        width: '40px',
        height: '40px',
        background: 'rgba(217, 142, 95, 0.2)',
        borderRadius: '6px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '1.3rem',
        fontWeight: 700,
        color: '#D98E5F',
        margin: '0 auto 1rem',
    },
    stepTitle: {
        fontSize: '1.1rem',
        fontWeight: 600,
        marginBottom: '0.5rem',
    },
    stepDesc: {
        fontSize: '0.9rem',
        color: '#9FB0A5',
        lineHeight: 1.5,
    },
    primaryButton: {
        padding: '1rem 2rem',
        background: '#D98E5F',
        color: '#0E1512',
        border: 'none',
        borderRadius: '8px',
        fontSize: '1rem',
        fontWeight: 600,
        cursor: 'pointer',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        transition: 'all 0.3s ease',
        minWidth: '200px',
    },
    primaryButtonHover: {
        background: '#E8C468',
        transform: 'translateY(-2px)',
        boxShadow: '0 4px 12px rgba(232, 196, 104, 0.2)',
    },
    secondaryButton: {
        padding: '0.75rem 1.5rem',
        background: 'transparent',
        color: '#D98E5F',
        border: '1px solid rgba(217, 142, 95, 0.4)',
        borderRadius: '8px',
        fontSize: '0.9rem',
        fontWeight: 600,
        cursor: 'pointer',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        transition: 'all 0.3s ease',
    },
    searchSection: {
        background: 'rgba(15, 21, 18, 0.6)',
        border: '1px solid rgba(159, 176, 165, 0.2)',
        borderRadius: '12px',
        padding: '2rem',
    },
    inputWrapper: {
        marginBottom: '2rem',
    },
    label: {
        display: 'block',
        fontSize: '0.85rem',
        color: '#D98E5F',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '0.5rem',
        fontWeight: 600,
    },
    textarea: {
        width: '100%',
        minHeight: '100px',
        padding: '1rem',
        background: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(159, 176, 165, 0.2)',
        borderRadius: '8px',
        color: '#F4F3EF',
        fontFamily: 'inherit',
        fontSize: '1rem',
        resize: 'vertical',
        transition: 'all 0.2s',
    },
    examplesSection: {
        marginBottom: '2rem',
    },
    examplesLabel: {
        fontSize: '0.85rem',
        color: '#D98E5F',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '1rem',
        fontWeight: 600,
    },
    chipGroup: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.75rem',
    },
    chip: {
        padding: '0.75rem 1rem',
        background: 'rgba(95, 184, 154, 0.1)',
        color: '#5FB89A',
        border: '1px solid rgba(95, 184, 154, 0.3)',
        borderRadius: '24px',
        cursor: 'pointer',
        fontSize: '0.85rem',
        fontWeight: 500,
        transition: 'all 0.2s',
    },
    chipActive: {
        background: 'rgba(95, 184, 154, 0.3)',
        borderColor: '#5FB89A',
    },
    buttonGroup: {
        display: 'flex',
        justifyContent: 'center',
        gap: '1rem',
    },
    loadingContainer: {
        textAlign: 'center',
        padding: '2rem',
    },
    spinner: {
        width: '40px',
        height: '40px',
        border: '3px solid rgba(217, 142, 95, 0.3)',
        borderTop: '3px solid #D98E5F',
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
        margin: '0 auto 1rem',
    },
    resultsSection: {
        width: '100%',
    },
    resultsHeader: {
        marginBottom: '2rem',
        textAlign: 'center',
    },
    resultsTitle: {
        fontSize: '1.8rem',
        marginBottom: '0.5rem',
        fontWeight: 700,
    },
    resultsSubtitle: {
        color: '#9FB0A5',
        fontSize: '1rem',
    },
    cardsGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '1.5rem',
        marginBottom: '2rem',
    },
    card: {
        background: 'rgba(15, 21, 18, 0.8)',
        border: '1px solid rgba(159, 176, 165, 0.2)',
        borderRadius: '12px',
        padding: '1.5rem',
        transition: 'all 0.3s ease',
    },
    cardHeader: {
        marginBottom: '1.5rem',
    },
    datasetName: {
        fontSize: '1.3rem',
        fontWeight: 700,
        marginBottom: '0.3rem',
    },
    datasetSource: {
        fontSize: '0.9rem',
        color: '#9FB0A5',
    },
    cardSection: {
        marginBottom: '1.5rem',
    },
    sectionLabel: {
        fontSize: '0.75rem',
        color: '#D98E5F',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '0.5rem',
        fontWeight: 600,
    },
    relevanceScore: {
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        marginBottom: '1rem',
    },
    scoreBadge: {
        fontSize: '1.8rem',
        fontWeight: 800,
        color: '#E8C468',
        minWidth: '60px',
        textAlign: 'center',
    },
    scoreBar: {
        flex: 1,
        height: '8px',
        background: 'rgba(159, 176, 165, 0.2)',
        borderRadius: '4px',
        overflow: 'hidden',
    },
    scoreFill: {
        height: '100%',
        background: 'linear-gradient(90deg, #5FB89A, #E8C468)',
        borderRadius: '4px',
    },
    flag: {
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
        marginBottom: '1rem',
    },
    flagIcon: {
        width: '24px',
        height: '24px',
        borderRadius: '4px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.8rem',
        color: '#0E1512',
        flexShrink: 0,
        marginTop: '2px',
    },
    flagContent: {
        flex: 1,
    },
    flagTitle: {
        fontWeight: 600,
        marginBottom: '0.25rem',
        color: '#F4F3EF',
    },
    flagNote: {
        fontSize: '0.85rem',
        color: '#9FB0A5',
        lineHeight: 1.4,
    },
    cardLink: {
        color: '#E8C468',
        textDecoration: 'none',
        fontWeight: 600,
        fontSize: '0.9rem',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        transition: 'color 0.2s',
        cursor: 'pointer',
    },
    backButtonContainer: {
        textAlign: 'center',
    },
    footer: {
        textAlign: 'center',
        paddingTop: '2rem',
        borderTop: '1px solid rgba(159, 176, 165, 0.1)',
        color: '#9FB0A5',
        fontSize: '0.85rem',
        marginTop: 'auto',
    },
    footerText: {
        fontSize: '0.9rem',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        fontWeight: 600,
        color: '#D98E5F',
        marginBottom: '0.5rem',
    },
};
