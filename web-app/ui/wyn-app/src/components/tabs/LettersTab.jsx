export default function LettersTab() {
  const letters = [
    { year: 2025, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2025.pdf' },
    { year: 2024, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2024.pdf' },
    { year: 2023, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2023.pdf' },
    { year: 2022, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2022.pdf' },
    { year: 2021, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2021.pdf' },
    { year: 2020, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2020.pdf' },
    { year: 2019, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2019.pdf' },
    { year: 2018, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2018.pdf' },
    { year: 2017, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2017.pdf' },
    { year: 2016, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2016.pdf' },
    { year: 2015, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2015.pdf' },
    { year: 2014, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2014.pdf' },
    { year: 2013, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2013.pdf' },
    { year: 2012, url: 'https://wyn-associates.s3.amazonaws.com/public/letters/2012.pdf' },
  ];

  return (
    <>
      <h2 data-spotlight="letters-title">Letters</h2>
      <p data-spotlight="letters-description">Annual letters to partners and investors.</p>
      <div style={{ overflowX: 'auto' }}>
        <table data-spotlight="letters-table">
          <thead>
            <tr>
              <th>Year</th>
              <th>Document</th>
            </tr>
          </thead>
          <tbody>
            {letters.map(({ year, url }) => (
              <tr key={year}>
                <td>{year}</td>
                <td>
                  <a href={url} target="_blank" rel="noopener noreferrer" data-spotlight={`letters-${year}`}>
                    Annual Letter {year}
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
